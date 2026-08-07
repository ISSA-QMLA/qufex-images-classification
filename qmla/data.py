"""Galaxy Zoo 2 download, preprocessing, and memory-mapped dataset support."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
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
        self.output_dir = config.paths.processed_dir

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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame = self._read_and_label()
        image_index = self._image_index()
        frame["image_path"] = frame["asset_id"].astype(str).map(image_index)
        missing_images = int(frame["image_path"].isna().sum())
        frame = frame.dropna(subset=["image_path"]).reset_index(drop=True)
        if frame.empty:
            raise RuntimeError("No clean labelled images remained after joining the official data products")

        split_frames = self._split(frame)
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, split_frame in split_frames.items():
            arrays[name] = self._write_split(name, split_frame.reset_index(drop=True))
        mean, std = self._training_statistics(arrays["train"][0])

        class_names = self.config.evaluation.class_names
        metadata: dict[str, Any] = {
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
            }
        (self.output_dir / "dataset_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self.config.save_resolved(self.output_dir)
        print(json.dumps(metadata, indent=2))
        return metadata


class GalaxyDataset(TorchDataset):  # type: ignore[misc]
    """Memory-mapped processed split with lightweight tensor augmentation."""

    def __init__(self, processed_dir: Path, split: str, *, augment: bool = False) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for GalaxyDataset. Install the cluster-appropriate PyTorch build first."
            )
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.images = np.load(processed_dir / f"{split}_images.npy", mmap_mode="r")
        self.labels = np.load(processed_dir / f"{split}_labels.npy", mmap_mode="r")
        metadata = json.loads((processed_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
        self.mean = torch.tensor(metadata["normalization_mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(metadata["normalization_std"], dtype=torch.float32).view(3, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        # Copy removes the read-only memmap warning and lets augmentation modify safely.
        image = torch.from_numpy(np.asarray(self.images[index]).copy()).permute(2, 0, 1).float().div_(255.0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(1,))
            image = torch.rot90(image, int(torch.randint(0, 4, ()).item()), dims=(1, 2))
        image = (image - self.mean) / self.std
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, label


def iter_manifest_ids(processed_dir: Path, split: str) -> Iterator[str]:
    """Yield object IDs without loading a full manifest; useful for leakage checks."""

    frame = pd.read_csv(processed_dir / f"{split}_manifest.csv", usecols=["dr7objid"], dtype="string")
    yield from frame["dr7objid"].astype(str)
