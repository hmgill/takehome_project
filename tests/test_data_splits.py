from types import SimpleNamespace

import numpy as np
import pytest

import data_splits
from data_splits import SPLIT_ORDER, build_dataset_splits, load_dataset_splits


def unique_image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (28, 28), dtype=np.uint8)


@pytest.fixture
def pooled():
    """
    60 unique images across official splits, plus planted duplicates:

    * image 0 copied into val and test (cross-split, same label)
    * image 1 copied twice within train (within-split, same label)
    * image 2 copied into test with the opposite label (label conflict)
    """

    images = [unique_image(i) for i in range(60)]
    labels = [i % 2 for i in range(60)]
    source_split = ["train"] * 40 + ["val"] * 10 + ["test"] * 10

    extras = [
        (images[0], labels[0], "val"),
        (images[0], labels[0], "test"),
        (images[1], labels[1], "train"),
        (images[1], labels[1], "train"),
        (images[2], 1 - labels[2], "test"),
    ]

    for image, label, split in extras:
        images.append(image.copy())
        labels.append(label)
        source_split.append(split)

    source_split = np.array(source_split, dtype=object)
    source_index = np.zeros(len(labels), dtype=np.int64)
    for split in SPLIT_ORDER:
        mask = source_split == split
        source_index[mask] = np.arange(mask.sum())

    return np.stack(images), np.array(labels), source_split, source_index


def all_hashes(splits):
    return {s: list(splits.get(s).image_hash) for s in SPLIT_ORDER}


def test_no_duplicates_within_or_across_splits(pooled):
    splits = build_dataset_splits(*pooled)
    hashes = all_hashes(splits)

    for split in SPLIT_ORDER:
        assert len(hashes[split]) == len(set(hashes[split]))

    assert not set(hashes["train"]) & set(hashes["val"])
    assert not set(hashes["train"]) & set(hashes["test"])
    assert not set(hashes["val"]) & set(hashes["test"])


def test_report_counts_and_conflict_drop(pooled):
    splits = build_dataset_splits(*pooled, label_conflict_policy="drop")
    report = splits.report

    assert report["input_records"] == 65
    assert report["unique_images"] == 60
    assert report["duplicate_groups"] == 3
    assert report["redundant_copies_removed"] == 5
    assert report["label_conflict_groups"] == 1
    # The conflicting image 2 is removed entirely.
    assert report["retained_records"] == 59
    assert sum(report["split_sizes"].values()) == 59


def test_keep_first_retains_conflicting_image(pooled):
    splits = build_dataset_splits(*pooled, label_conflict_policy="keep_first")
    assert splits.report["retained_records"] == 60


def test_first_occurrence_provenance_is_kept(pooled):
    images, *_ = pooled
    splits = build_dataset_splits(*pooled)
    target = data_splits.calculate_image_hash(images[0])
    manifest = splits.manifest()
    row = manifest[manifest.image_hash == target]

    assert len(row) == 1
    assert row.source_split.iloc[0] == "train"
    assert row.source_index.iloc[0] == 0


def test_split_is_deterministic(pooled):
    first = build_dataset_splits(*pooled, random_seed=7)
    second = build_dataset_splits(*pooled, random_seed=7)
    assert first.report["split_fingerprint"] == second.report["split_fingerprint"]


def test_official_mode_returns_source_splits(pooled):
    splits = build_dataset_splits(*pooled, deduplicate_images=False)
    assert len(splits.train) == 42
    assert len(splits.val) == 11
    assert len(splits.test) == 12


def test_invalid_policy_rejected(pooled):
    with pytest.raises(ValueError):
        build_dataset_splits(*pooled, label_conflict_policy="majority")


def test_load_from_npz(pooled, tmp_path):
    images, labels, source_split, _ = pooled
    arrays = {}
    for split in SPLIT_ORDER:
        mask = source_split == split
        arrays[f"{split}_images"] = images[mask]
        arrays[f"{split}_labels"] = labels[mask].reshape(-1, 1)

    path = tmp_path / "toy.npz"
    np.savez(path, **arrays)

    settings = SimpleNamespace(
        deduplicate=True,
        label_conflict_policy="drop",
        preserve_source_proportions=True,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    splits = load_dataset_splits(path, settings=settings, random_seed=42)
    hashes = all_hashes(splits)

    assert sum(len(h) for h in hashes.values()) == 59
    assert len(set().union(*map(set, hashes.values()))) == 59
