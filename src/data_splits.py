"""
Shared dataset ingestion: pool, deduplicate, then split.

Every script that trains or evaluates a model loads data through
``load_dataset_splits`` so that all of them see the same train/val/test
partition.

When ``[splits].deduplicate`` is enabled (the default), the official
MedMNIST train/val/test arrays are pooled into one set, exact duplicates
(SHA-256 over dtype, shape and pixels, see ``utils.calculate_image_hash``)
are collapsed to a single record, and the unique images are then
re-split with a stratified, seeded split. Because duplicates are removed
*before* splitting, no image can appear in more than one split, and no
image is repeated inside a split.

Duplicate groups whose copies carry different labels are handled by
``[splits].label_conflict_policy``:

* ``"drop"`` (default): remove every copy; the true label is ambiguous.
* ``"keep_first"``: keep the first copy and its label.

Each retained record keeps its source coordinates (``source_split``,
``source_index``) so it can still be joined to the DuckDB metadata and
looked up in the NPZ.

When ``deduplicate`` is disabled, the official splits are returned
unchanged, which keeps results comparable to published benchmarks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from utils import calculate_image_hash, get_split_arrays

SPLIT_ORDER = ("train", "val", "test")

LABEL_CONFLICT_POLICIES = {"drop", "keep_first"}


@dataclass(frozen=True)
class SplitData:
    """Images and provenance for one assigned split."""

    images: np.ndarray
    labels: np.ndarray
    image_hash: np.ndarray
    source_split: np.ndarray
    source_index: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class DatasetSplits:
    """The train/val/test partition shared by every modeling script."""

    train: SplitData
    val: SplitData
    test: SplitData
    report: dict = field(default_factory=dict)

    def get(self, split: str) -> SplitData:
        if split not in SPLIT_ORDER:
            raise ValueError(f"Unexpected split: {split}")

        return getattr(self, split)

    def manifest(self) -> pd.DataFrame:
        """One row per retained image with its assigned split."""

        frames = []

        for split in SPLIT_ORDER:
            data = self.get(split)
            frames.append(
                pd.DataFrame(
                    {
                        "assigned_split": split,
                        "assigned_index": np.arange(len(data)),
                        "source_split": data.source_split,
                        "source_index": data.source_index,
                        "label": data.labels,
                        "image_hash": data.image_hash,
                    }
                )
            )

        return pd.concat(frames, ignore_index=True)


def _pool_official_splits(
    dataset: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate the official splits, recording where each image came from."""

    images, labels, source_split, source_index = [], [], [], []

    for split in SPLIT_ORDER:
        split_images, split_labels = get_split_arrays(dataset, split)
        images.append(split_images)
        labels.append(split_labels)
        source_split.append(np.full(len(split_labels), split, dtype=object))
        source_index.append(np.arange(len(split_labels), dtype=np.int64))

    return (
        np.concatenate(images),
        np.concatenate(labels),
        np.concatenate(source_split),
        np.concatenate(source_index),
    )


def deduplicate(
    image_hash: np.ndarray,
    labels: np.ndarray,
    label_conflict_policy: str = "drop",
) -> tuple[np.ndarray, dict]:
    """
    Return indices of records to keep, one per unique image hash.

    The first occurrence (in train, val, test pool order) is kept.
    """

    if label_conflict_policy not in LABEL_CONFLICT_POLICIES:
        raise ValueError(
            f"label_conflict_policy must be one of "
            f"{sorted(LABEL_CONFLICT_POLICIES)}, got {label_conflict_policy!r}"
        )

    frame = pd.DataFrame(
        {
            "position": np.arange(len(labels)),
            "image_hash": image_hash,
            "label": labels,
        }
    )

    group_size = frame.groupby("image_hash")["image_hash"].transform("size")
    label_count = frame.groupby("image_hash")["label"].transform("nunique")

    conflicting = label_count > 1
    first = ~frame.duplicated("image_hash", keep="first")

    keep = first
    if label_conflict_policy == "drop":
        keep = keep & ~conflicting

    report = {
        "input_records": int(len(frame)),
        "unique_images": int(frame["image_hash"].nunique()),
        "duplicate_groups": int(frame.loc[group_size > 1, "image_hash"].nunique()),
        "redundant_copies_removed": int((~first).sum()),
        "label_conflict_groups": int(frame.loc[conflicting, "image_hash"].nunique()),
        "label_conflict_policy": label_conflict_policy,
        "label_conflict_records_dropped": (
            int((first & conflicting).sum()) if label_conflict_policy == "drop" else 0
        ),
    }

    report["retained_records"] = int(keep.sum())

    return frame.loc[keep, "position"].to_numpy(), report


def _split_fractions(
    source_split: np.ndarray,
    preserve_source_proportions: bool,
    val_fraction: float,
    test_fraction: float,
) -> tuple[float, float]:
    if preserve_source_proportions:
        total = len(source_split)
        val_fraction = float(np.sum(source_split == "val") / total)
        test_fraction = float(np.sum(source_split == "test") / total)

    if not (0 < val_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")

    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be below 1")

    return val_fraction, test_fraction


def _subset(
    positions: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    image_hash: np.ndarray,
    source_split: np.ndarray,
    source_index: np.ndarray,
) -> SplitData:
    # Sort by pool position so each split has a stable, readable order.
    positions = np.sort(positions)

    return SplitData(
        images=images[positions],
        labels=labels[positions],
        image_hash=image_hash[positions],
        source_split=source_split[positions],
        source_index=source_index[positions],
    )


def split_fingerprint(splits: DatasetSplits) -> str:
    """Short digest identifying exactly which images went to which split."""

    hasher = hashlib.sha256()

    for split in SPLIT_ORDER:
        hasher.update(split.encode("utf-8"))
        for value in sorted(splits.get(split).image_hash):
            hasher.update(str(value).encode("utf-8"))

    return hasher.hexdigest()[:16]


def build_dataset_splits(
    images: np.ndarray,
    labels: np.ndarray,
    source_split: np.ndarray,
    source_index: np.ndarray,
    deduplicate_images: bool = True,
    label_conflict_policy: str = "drop",
    preserve_source_proportions: bool = True,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    random_seed: int = 42,
) -> DatasetSplits:
    """Deduplicate pooled records, then produce the train/val/test partition."""

    labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    image_hash = np.array([calculate_image_hash(image) for image in images])

    if not deduplicate_images:
        parts = {
            split: _subset(
                np.flatnonzero(source_split == split),
                images,
                labels,
                image_hash,
                source_split,
                source_index,
            )
            for split in SPLIT_ORDER
        }
        splits = DatasetSplits(**parts, report={"strategy": "official"})
        splits.report["split_sizes"] = {s: len(parts[s]) for s in SPLIT_ORDER}
        splits.report["split_fingerprint"] = split_fingerprint(splits)
        return splits

    keep, report = deduplicate(image_hash, labels, label_conflict_policy)

    val_fraction, test_fraction = _split_fractions(
        source_split,
        preserve_source_proportions,
        val_fraction,
        test_fraction,
    )

    if len(keep) == 0:
        # Everything was removed (e.g. only conflicting duplicates).
        development = test_positions = train_positions = val_positions = keep
    else:
        development, test_positions = train_test_split(
            keep,
            test_size=test_fraction,
            stratify=labels[keep],
            random_state=random_seed,
        )

        train_positions, val_positions = train_test_split(
            development,
            test_size=val_fraction / (1 - test_fraction),
            stratify=labels[development],
            random_state=random_seed,
        )

    common = (images, labels, image_hash, source_split, source_index)

    splits = DatasetSplits(
        train=_subset(train_positions, *common),
        val=_subset(val_positions, *common),
        test=_subset(test_positions, *common),
        report={
            "strategy": "deduplicate_then_split",
            **report,
            "random_seed": int(random_seed),
            "val_fraction": round(val_fraction, 6),
            "test_fraction": round(test_fraction, 6),
            "preserve_source_proportions": bool(preserve_source_proportions),
        },
    )

    # Defensive check: the whole point of this module.
    seen: set[str] = set()
    for split in SPLIT_ORDER:
        hashes = set(splits.get(split).image_hash)
        if len(hashes) != len(splits.get(split)) or seen & hashes:
            raise RuntimeError("Duplicate images survived deduplication")
        seen |= hashes

    splits.report["split_sizes"] = {s: len(splits.get(s)) for s in SPLIT_ORDER}
    splits.report["split_fingerprint"] = split_fingerprint(splits)

    return splits


def load_dataset_splits(
    dataset_path: Path,
    settings=None,
    random_seed: int | None = None,
) -> DatasetSplits:
    """
    Load the NPZ and return the shared train/val/test partition.

    ``settings`` defaults to ``CONFIG.splits`` and ``random_seed`` to
    ``CONFIG.project.random_seed``.
    """

    if settings is None or random_seed is None:
        from project_config import CONFIG

        settings = settings or CONFIG.splits
        random_seed = CONFIG.project.random_seed if random_seed is None else random_seed

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as dataset:
        pooled = _pool_official_splits(dataset)

    splits = build_dataset_splits(
        *pooled,
        deduplicate_images=settings.deduplicate,
        label_conflict_policy=settings.label_conflict_policy,
        preserve_source_proportions=settings.preserve_source_proportions,
        val_fraction=settings.val_fraction,
        test_fraction=settings.test_fraction,
        random_seed=random_seed,
    )

    report = splits.report

    if report["strategy"] == "official":
        logger.warning(
            "Deduplication disabled: using official splits as-is "
            "(cross-split duplicates may be present)"
        )
    else:
        logger.success(
            "Deduplicated {} records to {} unique images "
            "({} duplicate groups, {} redundant copies removed, "
            "{} label-conflict groups, policy='{}')",
            report["input_records"],
            report["unique_images"],
            report["duplicate_groups"],
            report["redundant_copies_removed"],
            report["label_conflict_groups"],
            report["label_conflict_policy"],
        )

    logger.success(
        "Split sizes: train={}, val={}, test={} (fingerprint {})",
        len(splits.train),
        len(splits.val),
        len(splits.test),
        report["split_fingerprint"],
    )

    return splits


def save_split_manifest(splits: DatasetSplits, output_dir: Path) -> list[Path]:
    """Write the per-image split assignment and the dedup report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "split_manifest.csv"
    report_path = output_dir / "split_report.json"

    splits.manifest().to_csv(manifest_path, index=False)
    report_path.write_text(json.dumps(splits.report, indent=2) + "\n", encoding="utf-8")

    logger.info("Saved split manifest to {}", manifest_path)

    return [manifest_path, report_path]
