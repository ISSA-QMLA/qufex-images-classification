from pathlib import Path
import json
import numpy as np
import pandas as pd

SOURCE = Path("/data/qmla/famato/data/processed")
DEST = Path("/data/qmla/famato/data/processed_smoke")

N_PER_CLASS = {
    "train": 5000,
    "validation": 300,
    "test": 300,
}

DEST.mkdir(parents=True, exist_ok=True)

for split, n_per_class in N_PER_CLASS.items():
    images = np.load(SOURCE / f"{split}_images.npy", mmap_mode="r")
    labels = np.load(SOURCE / f"{split}_labels.npy", mmap_mode="r")
    manifest = pd.read_csv(SOURCE / f"{split}_manifest.csv")

    indices = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        indices.extend(class_indices[:n_per_class])

    indices = np.array(sorted(indices))

    np.save(
        DEST / f"{split}_images.npy",
        np.asarray(images[indices]),
    )
    np.save(
        DEST / f"{split}_labels.npy",
        np.asarray(labels[indices])
    )

    manifest.iloc[indices].to_csv(
        DEST / f"{split}_manifest.csv",
        index=False,
    )

metadata = json.loads(
    (SOURCE / "dataset_metadata.json").read_text()
)

for split in N_PER_CLASS:
    labels = np.load(DEST / f"{split}_labels.npy")
    counts = np.bincount(labels, minlength=3)

    metadata['splits'][split]['samples'] = len(labels)
    metadata['splits'][split]["class_counts"] = {
        metadata["class_names"][i]: int(counts[i])
        for i in range(3)
    }

(DEST / "dataset_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

