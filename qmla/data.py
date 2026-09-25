"""Galaxy Zoo 2 download, preprocessing, and memory-mapped dataset support."""

from __future__ import annotations

import hashlib
import json
import zipfile
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split

from qmla.config import AppConfig

try:  # Data extraction and preprocessing deliberately work without PyTorch.
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised on data-only HPC nodes
    torch = None
    TorchDataset = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class RemoteResource:
    filename: str
    url: str
    md5: str | None = None


RESOURCES = (
    RemoteResource(
        "images_gz2.zip",
        "https://zenodo.org/records/3565489/files/images_gz2.zip?download=1",
        "bc647032d31e50c798770cf4430525c7",
    ),
    RemoteResource(
        "gz2_filename_mapping.csv",
        "https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1",
        "7e28465e6dfbf96c0d828c595a5bbd80",
    ),
    RemoteResource(
        "gz2_hart16.csv.gz",
        "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz",
    ),
)


def file_digest(path: Path, algorithm: str = "sha256", chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Safely extract useful files while ignoring macOS archive metadata."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            archive_path = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                "__MACOSX" in archive_path.parts
                or archive_path.name == ".DS_Store"
                or archive_path.name.startswith("._")
            ):
                continue
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            bundle.extract(member, root)


class GalaxyZooDownloader:
    """Resumable downloader for the official GZ2 images, mapping, and labels."""

    def __init__(self, raw_dir: Path, *, timeout_seconds: int = 120) -> None:
        self.raw_dir = raw_dir
        self.timeout_seconds = timeout_seconds

    def _download(self, resource: RemoteResource) -> Path:
        destination = self.raw_dir / resource.filename
        partial = destination.with_suffix(destination.suffix + ".part")
        if destination.exists() and self._valid(destination, resource):
            print(f"Using existing verified file: {destination}")
            return destination

        headers: dict[str, str] = {}
        mode = "wb"
        downloaded = 0
        if partial.exists():
            downloaded = partial.stat().st_size
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"

        print(f"Downloading {resource.url} -> {destination}")
        with requests.get(
            resource.url,
            headers=headers,
            stream=True,
            timeout=(30, self.timeout_seconds),
            allow_redirects=True,
        ) as response:
            if response.status_code == 416 and partial.exists():
                partial.replace(destination)
            else:
                response.raise_for_status()
                if downloaded and response.status_code != 206:
                    mode = "wb"
                    downloaded = 0
                total = int(response.headers.get("Content-Length", 0)) + downloaded
                written = downloaded
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if total:
                            print(f"\r  {written / (1024**2):,.1f}/{total / (1024**2):,.1f} MiB", end="")
                if total:
                    print()
                partial.replace(destination)

        if not self._valid(destination, resource):
            raise RuntimeError(f"Checksum validation failed for {destination}")
        return destination

    @staticmethod
    def _valid(path: Path, resource: RemoteResource) -> bool:
        return resource.md5 is None or file_digest(path, "md5") == resource.md5

    def run(self, *, extract_images: bool = True) -> dict[str, Any]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        downloaded = {resource.filename: self._download(resource) for resource in RESOURCES}
        images_dir = self.raw_dir / "images"
        if extract_images:
            marker = images_dir / ".extraction_complete"
            if not marker.exists():
                print(f"Extracting {downloaded['images_gz2.zip']} -> {images_dir}")
                safe_extract_zip(downloaded["images_gz2.zip"], images_dir)
                marker.touch()
            else:
                print(f"Using previously extracted images: {images_dir}")

        manifest = {
            "source": "Galaxy Zoo 2: Images from Original Sample, Zenodo 3565489",
            "files": {
                name: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_digest(path),
                }
                for name, path in downloaded.items()
            },
        }
        (self.raw_dir / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest


LABEL_COLUMNS = {
    "smooth": "t01_smooth_or_features_a01_smooth_flag",
    "barred": "t03_bar_a06_bar_flag",
    "unbarred": "t03_bar_a07_no_bar_flag",
    "spiral": "t04_spiral_a08_spiral_flag",
}


class GalaxyZooPreprocessor:
    """Build clean three-class, leakage-free, memory-mapped data splits."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.raw_dir = config.paths.raw_dir
        self.output_dir = config.cache_dir

    def _read_and_label(self) -> pd.DataFrame:
        mapping_path = self.raw_dir / "gz2_filename_mapping.csv"
        catalog_path = self.raw_dir / "gz2_hart16.csv.gz"
        if not mapping_path.is_file() or not catalog_path.is_file():
            raise FileNotFoundError("Raw mapping/catalog files are missing; run scripts/extract_data.py first")

        mapping = pd.read_csv(
            mapping_path,
            dtype={"objid": "string", "sample": "string", "asset_id": "Int64"},
        )
        required_mapping = {"objid", "sample", "asset_id"}
        if missing := required_mapping - set(mapping.columns):
            raise ValueError(f"Mapping is missing columns: {sorted(missing)}")
        original = mapping["sample"].str.lower().eq("original")
        if original.any():
            mapping = mapping.loc[original]
        mapping = mapping.dropna(subset=["objid", "asset_id"]).drop_duplicates("objid", keep="first")
        mapping = mapping.rename(columns={"objid": "dr7objid"})

        wanted = ["dr7objid", *LABEL_COLUMNS.values()]
        catalog = pd.read_csv(catalog_path, usecols=wanted, dtype={"dr7objid": "string"})
        missing = set(wanted) - set(catalog.columns)
        if missing:
            raise ValueError(f"Hart catalogue is missing columns: {sorted(missing)}")
        catalog = catalog.drop_duplicates("dr7objid", keep="first")
        frame = mapping.merge(catalog, on="dr7objid", how="inner", validate="one_to_one")

        smooth = frame[LABEL_COLUMNS["smooth"]].eq(1)
        spiral = frame[LABEL_COLUMNS["spiral"]].eq(1)
        unbarred = spiral & frame[LABEL_COLUMNS["unbarred"]].eq(1)
        barred = spiral & frame[LABEL_COLUMNS["barred"]].eq(1)
        membership = np.column_stack((smooth.to_numpy(), unbarred.to_numpy(), barred.to_numpy()))
        valid = membership.sum(axis=1) == 1
        frame = frame.loc[valid].copy()
        frame["label"] = membership[valid].argmax(axis=1).astype(np.int64)
        return frame

    def _image_index(self) -> dict[str, Path]:
        root = self.raw_dir / "images"
        if not root.is_dir():
            raise FileNotFoundError(f"Extracted image directory does not exist: {root}")
        print("Indexing extracted JPEG files...")
        return {path.stem: path for path in root.rglob("*.jpg")}

    def _split(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        train, temporary = train_test_split(
            frame,
            train_size=self.config.data.train_fraction,
            random_state=self.config.data.seed,
            stratify=frame["label"],
        )
        validation_share = self.config.data.validation_fraction / (
            self.config.data.validation_fraction + self.config.data.test_fraction
        )
        validation, test = train_test_split(
            temporary,
            train_size=validation_share,
            random_state=self.config.data.seed,
            stratify=temporary["label"],
        )
        return {"train": train, "validation": validation, "test": test}

    def _subset(self, frame: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Largest-remainder allocation, retaining every class even for tiny subsets."""
        if not limit or limit >= len(frame):
            return frame.reset_index(drop=True)
        rng = np.random.default_rng(self.config.data.subset_seed)
        counts = frame["label"].value_counts().sort_index()
        quotas = counts.to_numpy() * limit / len(frame)
        allocation = np.maximum(np.floor(quotas).astype(int), 1)
        while allocation.sum() > limit:
            candidates = np.where(allocation > 1, allocation - quotas, -np.inf)
            allocation[int(candidates.argmax())] -= 1
        while allocation.sum() < limit:
            candidates = np.where(allocation < counts.to_numpy(), quotas - allocation, -np.inf)
            allocation[int(candidates.argmax())] += 1
        pieces = []
        for label, amount in zip(counts.index, allocation):
            group = frame.loc[frame["label"] == label]
            pieces.append(group.iloc[rng.permutation(len(group))[:amount]])
        result = pd.concat(pieces)
        return result.iloc[rng.permutation(len(result))].reset_index(drop=True)

    def _write_split(self, name: str, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        size = self.config.data.image_size
        images = np.lib.format.open_memmap(
            self.output_dir / f"{name}_images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(frame), size, size, 3),
        )
        labels = np.lib.format.open_memmap(
            self.output_dir / f"{name}_labels.npy",
            mode="w+",
            dtype=np.int64,
            shape=(len(frame),),
        )
        for index, row in enumerate(frame.itertuples(index=False)):
            with Image.open(row.image_path) as image:
                rgb = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
                images[index] = np.asarray(rgb, dtype=np.uint8)
            labels[index] = int(row.label)
            if (index + 1) % 1000 == 0 or index + 1 == len(frame):
                print(f"\rWriting {name}: {index + 1:,}/{len(frame):,}", end="")
        print()
        images.flush()
        labels.flush()
        output_columns = ["dr7objid", "asset_id", "sample", "label", "image_path"]
        frame.loc[:, output_columns].to_csv(self.output_dir / f"{name}_manifest.csv", index=False)
        return images, labels

    def _training_statistics(self, images: np.ndarray) -> tuple[list[float], list[float]]:
        total = np.zeros(3, dtype=np.float64)
        total_squares = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        step = self.config.data.preprocessing_batch_size
        for start in range(0, len(images), step):
            batch = np.asarray(images[start : start + step], dtype=np.float64) / 255.0
            total += batch.sum(axis=(0, 1, 2))
            total_squares += np.square(batch).sum(axis=(0, 1, 2))
            pixel_count += batch.shape[0] * batch.shape[1] * batch.shape[2]
        mean = total / pixel_count
        variance = np.maximum(total_squares / pixel_count - np.square(mean), 1e-12)
        return mean.tolist(), np.sqrt(variance).tolist()

    def run(self) -> dict[str, Any]:
        destination = self.config.cache_dir
        if destination.exists():
            metadata = validate_cache(self.config)
            print(f"Using verified cache: {destination}")
            return metadata
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
        self.output_dir.mkdir()
        frame = self._read_and_label()
        image_index = self._image_index()
        frame["image_path"] = frame["asset_id"].astype(str).map(image_index)
        missing_images = int(frame["image_path"].isna().sum())
        frame = frame.dropna(subset=["image_path"]).reset_index(drop=True)
        if frame.empty:
            raise RuntimeError("No clean labelled images remained after joining the official data products")

        split_frames = self._split(frame)
        master_ids = {name: identity_digest(split["dr7objid"].astype(str)) for name, split in split_frames.items()}
        split_frames = {name: self._subset(split, getattr(self.config.data, f"{name}_limit")) for name, split in split_frames.items()}
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, split_frame in split_frames.items():
            arrays[name] = self._write_split(name, split_frame.reset_index(drop=True))
        mean, std = self._training_statistics(arrays["train"][0])

        class_names = self.config.evaluation.class_names
        metadata: dict[str, Any] = {
            "format_version": 2,
            "data_signature": self.config.data_signature(),
            "master_ids": master_ids,
            "sources": {name: file_digest(self.raw_dir / name) for name in ("gz2_filename_mapping.csv", "gz2_hart16.csv.gz")},
            "image_size": self.config.data.image_size,
            "class_names": list(class_names),
            "class_to_index": {name: index for index, name in enumerate(class_names)},
            "normalization_mean": mean,
            "normalization_std": std,
            "missing_images_dropped": missing_images,
            "splits": {},
        }
        for name, (_, labels) in arrays.items():
            counts = np.bincount(np.asarray(labels), minlength=len(class_names))
            metadata["splits"][name] = {
                "samples": int(len(labels)),
                "class_counts": {class_names[i]: int(counts[i]) for i in range(len(class_names))},
                "ids_sha256": identity_digest(split_frames[name]["dr7objid"].astype(str)),
            }
        metadata["files"] = {path.name: {"sha256": file_digest(path), "size_bytes": path.stat().st_size}
                             for path in self.output_dir.iterdir() if path.suffix in {".npy", ".csv"}}
        metadata["dataset_id"] = hashlib.sha256(json.dumps(
            {key: metadata[key] for key in ("data_signature", "sources", "master_ids", "files")}, sort_keys=True).encode()).hexdigest()
        (self.output_dir / "dataset_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self.config.save_resolved(self.output_dir)
        for images_array, labels_array in arrays.values():
            images_array._mmap.close()
            labels_array._mmap.close()
        # Publish only complete caches; failed builds remain isolated for inspection.
        os.rename(self.output_dir, destination)
        self.output_dir = destination
        print(json.dumps(metadata, indent=2))
        return metadata


class GalaxyDataset(TorchDataset):  # type: ignore[misc]
    """Memory-mapped processed split with lightweight tensor augmentation."""

    def __init__(self, processed_dir: Path, split: str, *, augment: bool = False,
                 augmentation_seed: int | None = None) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for GalaxyDataset. Install the cluster-appropriate PyTorch build first."
            )
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.processed_dir = processed_dir
        self.split = split
        self._open_arrays()
        metadata = json.loads((processed_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
        self.mean = torch.tensor(metadata["normalization_mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(metadata["normalization_std"], dtype=torch.float32).view(3, 1, 1)
        self.augment = augment
        self.augmentation_seed = augmentation_seed
        self.epoch = 0

    def _open_arrays(self) -> None:
        self.images = np.load(self.processed_dir / f"{self.split}_images.npy", mmap_mode="r")
        self.labels = np.load(self.processed_dir / f"{self.split}_labels.npy", mmap_mode="r")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("images")
        state.pop("labels")
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open_arrays()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        # Copy removes the read-only memmap warning and lets augmentation modify safely.
        image = torch.from_numpy(np.asarray(self.images[index]).copy()).permute(2, 0, 1).float().div_(255.0)
        if self.augment:
            generator = None
            if self.augmentation_seed is not None:
                generator = torch.Generator().manual_seed(
                    (self.augmentation_seed + 1000003 * self.epoch + 9176 * index) % (2**63 - 1))
            if torch.rand((), generator=generator) < 0.5:
                image = torch.flip(image, dims=(2,))
            if torch.rand((), generator=generator) < 0.5:
                image = torch.flip(image, dims=(1,))
            image = torch.rot90(image, int(torch.randint(0, 4, (), generator=generator).item()), dims=(1, 2))
        image = (image - self.mean) / self.std
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, label


def iter_manifest_ids(processed_dir: Path, split: str) -> Iterator[str]:
    """Yield object IDs without loading a full manifest; useful for leakage checks."""

    frame = pd.read_csv(processed_dir / f"{split}_manifest.csv", usecols=["dr7objid"], dtype="string")
    yield from frame["dr7objid"].astype(str)


def identity_digest(ids: Any) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def validate_cache(config: AppConfig) -> dict[str, Any]:
    """Validate provenance, file integrity, shapes, labels and disjoint identities."""
    directory = config.cache_dir
    hint = f"Run uv run --no-sync python -m scripts.preprocess_data --config configs/experiments.toml --profile {config.run.profile}. Cache: {directory}"
    try:
        metadata = json.loads((directory / "dataset_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 2 or metadata.get("data_signature") != config.data_signature():
            raise ValueError("Dataset signature is incompatible")
        expected_id = hashlib.sha256(json.dumps(
            {key: metadata[key] for key in ("data_signature", "sources", "master_ids", "files")}, sort_keys=True).encode()).hexdigest()
        if metadata.get("dataset_id") != expected_id:
            raise ValueError("Dataset provenance is inconsistent")
        for name, expected in metadata["sources"].items():
            path = config.paths.raw_dir / name
            if path.exists() and file_digest(path) != expected:
                raise ValueError(f"Raw source changed: {name}; choose a new processed_dir to rebuild")
        all_ids: set[str] = set()
        for split in ("train", "validation", "test"):
            for suffix in ("images.npy", "labels.npy", "manifest.csv"):
                name = f"{split}_{suffix}"
                path = directory / name
                record = metadata["files"][name]
                if path.stat().st_size != record["size_bytes"] or file_digest(path) != record["sha256"]:
                    raise ValueError(f"Cache file changed: {name}; use a new processed_dir to rebuild")
            images = np.load(directory / f"{split}_images.npy", mmap_mode="r")
            labels = np.load(directory / f"{split}_labels.npy", mmap_mode="r")
            manifest = pd.read_csv(directory / f"{split}_manifest.csv", dtype={"dr7objid": str})
            size = config.data.image_size
            if len(labels) == 0 or images.shape != (len(labels), size, size, 3) or images.dtype != np.uint8 or labels.dtype != np.int64:
                raise ValueError(f"Invalid {split} array dimensions/dtypes")
            if set(np.unique(labels)) != {0, 1, 2} or not np.array_equal(labels, manifest["label"].to_numpy()):
                raise ValueError(f"Invalid {split} labels")
            ids = manifest["dr7objid"].tolist()
            if len(set(ids)) != len(ids) or all_ids.intersection(ids):
                raise ValueError("Duplicate identities or split leakage")
            all_ids.update(ids)
            if metadata["splits"][split]["ids_sha256"] != identity_digest(ids) or metadata["splits"][split]["samples"] != len(labels):
                raise ValueError(f"Invalid {split} identity metadata")
        mean, std = np.asarray(metadata["normalization_mean"]), np.asarray(metadata["normalization_std"])
        if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid normalization statistics")
        return metadata
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Dataset validation failed: {exc}. {hint}") from exc
