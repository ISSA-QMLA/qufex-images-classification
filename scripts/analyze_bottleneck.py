"""Analyze saved QUFEX/CNN bottlenecks on the HPC; export small files for Jupyter.

Run from the repository root with ``python -m scripts.analyze_bottleneck --help``.
Feature caches stay on the compute host. No original-network training is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from qmla.config import config_from_dict
from qmla.data import GalaxyDataset
from qmla.model import GalaxyClassifier, residual_for_filters
from qmla.utils import configure_accelerator, resolve_device

MODELS = ("qufex", "cnn_replacement")
SPLITS = ("train", "validation", "test")
SHUFFLE_SEEDS = (42, 43, 44, 45, 46)
PROBE_C_VALUES = (0.01, 0.1, 1.0, 10.0)
FEATURE_KEYS = ("z", "b", "labels", "object_ids")


@dataclass
class AnalysisSettings:
    experiment: str
    checkpoint_root: Path
    processed_cache: Path | None
    output_dir: Path
    device: torch.device
    batch_size: int
    reuse_features: bool
    cpu_threads: int


def load_model(checkpoint_path, settings):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint["config"]
    raw["training"]["device"] = str(settings.device)
    config = config_from_dict(raw, root=checkpoint_path.parent)
    if settings.processed_cache is None:
        settings.processed_cache = config.cache_dir
    configure_accelerator(
        settings.device, deterministic=config.training.deterministic,
        cpu_threads=settings.cpu_threads,
    )
    model = GalaxyClassifier(config).to(settings.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    amp = config.training.amp and settings.device.type == "cuda"
    info = {
        "path": str(checkpoint_path), "dataset_id": checkpoint["dataset_id"],
        "epoch": checkpoint["epoch"], "seed": config.training.seeds[0], "amp": amp,
    }
    return model, config, amp, info


def feature_directory(settings, mode, split):
    return settings.output_dir / "features" / mode / split


def load_features(settings, mode, split):
    directory = feature_directory(settings, mode, split)
    return {key: np.load(directory / f"{key}.npy", mmap_mode="r") for key in FEATURE_KEYS}


@torch.inference_mode()
def extract_features(model, mode, split, amp, settings):
    """Write each batch directly to .npy memory maps, without concatenating a split."""
    directory = feature_directory(settings, mode, split)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = GalaxyDataset(settings.processed_cache, split, augment=False)
    loader = DataLoader(
        dataset, batch_size=settings.batch_size, shuffle=False, num_workers=0,
        pin_memory=settings.device.type == "cuda",
    )
    offset = 0
    for images, _ in loader:
        images = images.to(settings.device, non_blocking=True)
        with torch.autocast(device_type=settings.device.type, enabled=amp):
            z = model.compression(model.encoder(images))
            if model.qufex is not None:
                with torch.autocast(device_type=settings.device.type, enabled=False):
                    b = model.qufex(z.float())
            else:
                b = model.replacement(z)
            b = b.to(dtype=z.dtype)
        z_batch, b_batch = z.cpu().numpy(), b.cpu().numpy()
        if offset == 0:
            z_map = np.lib.format.open_memmap(
                directory / "z.npy", mode="w+", dtype=z_batch.dtype,
                shape=(len(dataset), *z_batch.shape[1:]),
            )
            b_map = np.lib.format.open_memmap(
                directory / "b.npy", mode="w+", dtype=b_batch.dtype,
                shape=(len(dataset), *b_batch.shape[1:]),
            )
        stop = offset + len(images)
        z_map[offset:stop] = z_batch
        b_map[offset:stop] = b_batch
        offset = stop
        if offset % (settings.batch_size * 100) == 0 or offset == len(dataset):
            print(f"  {mode}/{split}: {offset:,}/{len(dataset):,}", flush=True)
    z_map.flush()
    b_map.flush()
    del z_map, b_map
    np.save(directory / "labels.npy", np.asarray(dataset.labels))
    manifest = pd.read_csv(
        settings.processed_cache / f"{split}_manifest.csv",
        usecols=["dr7objid"], dtype={"dr7objid": str},
    )
    np.save(directory / "object_ids.npy", manifest["dr7objid"].to_numpy(dtype=str))


def get_features(model, mode, split, amp, settings):
    if not settings.reuse_features:
        extract_features(model, mode, split, amp, settings)
    return load_features(settings, mode, split)


def metric_values(labels, probabilities, class_names):
    class_indices = np.arange(len(class_names))
    per_class = f1_score(
        labels, probabilities.argmax(axis=1), labels=class_indices,
        average=None, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, probabilities.argmax(axis=1)),
        "macro_f1": f1_score(
            labels, probabilities.argmax(axis=1), labels=class_indices,
            average="macro", zero_division=0,
        ),
        "cross_entropy": log_loss(
            labels, probabilities.astype(np.float64), labels=class_indices
        ),
        **{f"f1_{name}": score for name, score in zip(class_names, per_class)},
    }


@torch.inference_mode()
def head_probabilities(model, features, amp, settings, scale=1.0, permutation=None):
    probabilities = []
    for start in range(0, len(features["labels"]), settings.batch_size):
        stop = start + settings.batch_size
        z = torch.from_numpy(np.array(features["z"][start:stop], copy=True)).to(settings.device)
        indices = slice(start, stop) if permutation is None else permutation[start:stop]
        b = torch.from_numpy(np.array(features["b"][indices], copy=True)).to(settings.device)
        with torch.autocast(device_type=settings.device.type, enabled=amp):
            residual = residual_for_filters(z, model.filters)
            combined = residual if scale == 0 else residual + scale * b
            hidden = model.post_quantum(combined)
            hidden = model.global_pool(hidden).flatten(1)
            logits = model.classifier(hidden)
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def intervention_rows(model, mode, features, amp, class_names, settings):
    variants = [
        ("original", 1.0, None, None),
        ("removed", 0.0, None, None),
        ("half_strength", 0.5, None, None),
    ]
    for seed in SHUFFLE_SEEDS:
        permutation = np.random.default_rng(seed).permutation(len(features["labels"]))
        variants.append(("shuffled", 1.0, permutation, seed))
    rows = []
    for name, scale, permutation, seed in variants:
        probabilities = head_probabilities(model, features, amp, settings, scale, permutation)
        scores = metric_values(features["labels"], probabilities, class_names)
        if name == "original":
            baseline = scores
        rows.append({
            "experiment": settings.experiment, "model": mode,
            "variant": name, "shuffle_seed": seed,
            **scores,
            **{f"delta_{metric}": value - baseline[metric] for metric, value in scores.items()},
        })
        print(f"  {name:14s} seed={str(seed):4s} macro-F1={scores['macro_f1']:.4f}")
    return rows

def probe_features(features, representation, filters):
    if representation == "compression":
        values = features["z"]
    elif representation == "branch":
        values = features["b"]
    else:
        residual = np.repeat(features["z"], filters, axis=1) if filters > 1 else features["z"]
        values = residual + features["b"]
    return values.reshape(len(values), -1).astype(np.float64)


def fit_probe(splits, mode, representation, filters, class_names, settings):
    x_train = probe_features(splits["train"], representation, filters)
    x_validation = probe_features(splits["validation"], representation, filters)
    y_train = splits["train"]["labels"]
    y_validation = splits["validation"]["labels"]
    best_score, best_probe, best_c = -np.inf, None, None
    selection_rows = []
    for c_value in sorted(PROBE_C_VALUES):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value, class_weight="balanced", max_iter=2000,
                solver="lbfgs",
            ),
        )
        probe.fit(x_train, y_train)
        score = f1_score(
            y_validation, probe.predict(x_validation),
            labels=np.arange(len(class_names)), average="macro", zero_division=0,
        )
        selection_rows.append({
            "experiment": settings.experiment, "model": mode,
            "representation": representation, "C": c_value,
            "validation_macro_f1": score,
        })
        if score > best_score:
            best_score, best_probe, best_c = score, probe, c_value
    x_test = probe_features(splits["test"], representation, filters)
    probabilities = best_probe.predict_proba(x_test)
    row = {
        "experiment": settings.experiment, "model": mode,
        "representation": representation,
        "selected_C": best_c, "validation_macro_f1": best_score,
        **metric_values(splits["test"]["labels"], probabilities, class_names),
    }
    return row, selection_rows

def summarize_interventions(interventions):
    columns = [
        c for c in interventions.columns
        if c not in ("experiment", "model", "variant", "shuffle_seed")
    ]
    summary = (
        interventions.groupby(["experiment", "model", "variant"], sort=False)[columns]
        .agg(["mean", "std"]).fillna(0)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def save_plots(intervention_summary, probes, experiment, output_dir):
    variants = ("removed", "half_strength", "shuffled")
    x = np.arange(len(variants))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, mode in enumerate(MODELS):
        frame = intervention_summary[intervention_summary["model"] == mode].set_index("variant").loc[list(variants)]
        ax.bar(
            x + (index - 0.5) * width,
            100 * frame["delta_macro_f1_mean"],
            width, label=mode,
            yerr=100 * frame["delta_macro_f1_std"], capsize=4,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ("Removed", "Half strength", "Shuffled"))
    ax.set_ylabel("Macro-F1 change from original (percentage points)")
    ax.set_title(f"{experiment}: dependence on the bottleneck branch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "intervention_macro_f1.png", dpi=180)
    plt.close(fig)
    representations = ("compression", "branch", "residual_sum")
    x = np.arange(len(representations))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, mode in enumerate(MODELS):
        frame = probes[probes["model"] == mode].set_index("representation").loc[list(representations)]
        ax.bar(x + (index - 0.5) * width, frame["macro_f1"], width, label=mode)
    ax.set_xticks(x, ("Compression z", "Branch b", "Residual z + b"))
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"{experiment}: linear accessibility of class information")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "probe_macro_f1.png", dpi=180)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="12560")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--processed-cache", type=Path,
        help="Exact cache directory, not its parent; defaults to the saved checkpoint's cache.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/bottleneck_analysis"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--reuse-features", action="store_true",
        help="Reuse complete caches from this script for the same inputs/settings; no freshness checks.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    experiment = f"full128_array_{args.run_id}"
    settings = AnalysisSettings(
        experiment=experiment,
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
        processed_cache=args.processed_cache.expanduser().resolve() if args.processed_cache else None,
        output_dir=args.output_root.expanduser().resolve() / experiment,
        device=resolve_device(args.device), batch_size=args.batch_size,
        reuse_features=args.reuse_features, cpu_threads=args.cpu_threads,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{experiment} | {settings.device} | batch size {settings.batch_size}", flush=True)
    all_interventions, probe_rows, selection_rows = [], [], []
    checkpoint_info = {}
    for mode in MODELS:
        checkpoint_path = settings.checkpoint_root / f"{experiment}_{mode}" / "best.pt"
        print(f"\nLoading {checkpoint_path}", flush=True)
        model, config, amp, info = load_model(checkpoint_path, settings)
        checkpoint_info[mode] = info
        print(f"Processed cache: {settings.processed_cache}", flush=True)
        for split in SPLITS:
            features = get_features(model, mode, split, amp, settings)
            if split == "test":
                all_interventions.extend(intervention_rows(
                    model, mode, features, amp, config.evaluation.class_names, settings
                ))
            del features
        filters = model.filters
        del model
        gc.collect()
        if settings.device.type == "cuda":
            torch.cuda.empty_cache()

        # Release the neural network before fitting one representation at a time.
        splits = {split: load_features(settings, mode, split) for split in SPLITS}
        with threadpool_limits(limits=settings.cpu_threads):
            for representation in ("compression", "branch", "residual_sum"):
                row, selections = fit_probe(
                    splits, mode, representation, filters,
                    config.evaluation.class_names, settings,
                )
                probe_rows.append(row)
                selection_rows.extend(selections)
                print(
                    f"  {mode}/{representation}: C={row['selected_C']:g}, "
                    f"test macro-F1={row['macro_f1']:.4f}", flush=True,
                )
        del splits, config
        gc.collect()

    interventions = pd.DataFrame(all_interventions)
    intervention_summary = summarize_interventions(interventions)
    probes = pd.DataFrame(probe_rows)
    compression_f1 = probes[probes["representation"] == "compression"].set_index("model")["macro_f1"]
    probes["delta_macro_f1_vs_compression"] = probes["macro_f1"] - probes["model"].map(compression_f1)
    for name, frame in (
        ("interventions", interventions), ("intervention_summary", intervention_summary),
        ("probes", probes), ("probe_selection", pd.DataFrame(selection_rows)),
    ):
        frame.to_csv(settings.output_dir / f"{name}.csv", index=False)
    metadata = {
        "experiment": experiment, "processed_cache": str(settings.processed_cache),
        "output_dir": str(settings.output_dir), "device": str(settings.device),
        "batch_size": settings.batch_size, "cpu_threads": settings.cpu_threads,
        "reuse_features": settings.reuse_features,
        "shuffle_seeds": list(SHUFFLE_SEEDS), "probe_c_values": list(PROBE_C_VALUES),
        "probe_class_weight": "balanced", "probe_max_iter": 2000,
        "probe_selection_metric": "validation_macro_f1", "checkpoints": checkpoint_info,
    }
    (settings.output_dir / "analysis_settings.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    save_plots(intervention_summary, probes, experiment, settings.output_dir)
    print(f"\nResults: {settings.output_dir}", flush=True)
    print("Copy the top-level CSV, JSON, and PNG files to your laptop; leave features/ on HPC.", flush=True)


if __name__ == "__main__":
    main()
