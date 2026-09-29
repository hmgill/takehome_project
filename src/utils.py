import hashlib

import numpy as np

VALID_SPLITS = {
    "train",
    "val",
    "test",
}


def normalize_grayscale_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize an image to a 2D grayscale array.
    """

    image = np.asarray(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[-1] == 1:
        return image[
            ...,
            0,
        ]

    raise ValueError(f"Expected a grayscale image, got shape {image.shape}")


def calculate_image_hash(
    image: np.ndarray,
) -> str:
    """
    Compute SHA-256 for exact duplicate detection.
    """

    image = normalize_grayscale_image(image)

    image = np.ascontiguousarray(image)

    hasher = hashlib.sha256()

    hasher.update(str(image.dtype).encode("utf-8"))

    hasher.update(str(image.shape).encode("utf-8"))

    hasher.update(image.tobytes())

    return hasher.hexdigest()


def calculate_image_statistics(
    image: np.ndarray,
) -> dict[str, float | int]:
    """
    Calculate image-level intensity statistics.
    """

    image = normalize_grayscale_image(image)

    if not np.issubdtype(
        image.dtype,
        np.number,
    ):
        raise ValueError(f"Image has non-numeric dtype: {image.dtype}")

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values")

    height, width = image.shape

    return {
        "width": int(width),
        "height": int(height),
        "mean_intensity": float(np.mean(image)),
        "std_intensity": float(np.std(image)),
        "min_intensity": float(np.min(image)),
        "max_intensity": float(np.max(image)),
    }


def get_image_from_npz(
    dataset: np.lib.npyio.NpzFile,
    split: str,
    split_index: int,
) -> np.ndarray:
    """
    Retrieve one image using its split and index.
    """

    if split not in VALID_SPLITS:
        raise ValueError(f"Unexpected split: {split}")

    image_key = f"{split}_images"

    if image_key not in dataset.files:
        raise KeyError(f"NPZ does not contain '{image_key}'")

    images = dataset[image_key]

    if not 0 <= split_index < len(images):
        raise IndexError(
            f"{split}[{split_index}] is outside " f"the range 0-{len(images) - 1}"
        )

    return normalize_grayscale_image(images[split_index])


def get_split_arrays(
    dataset: np.lib.npyio.NpzFile,
    split: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Return image and label arrays for one MedMNIST split.
    """

    if split not in VALID_SPLITS:
        raise ValueError(f"Unexpected split: {split}")

    image_key = f"{split}_images"

    label_key = f"{split}_labels"

    missing = {
        key
        for key in (
            image_key,
            label_key,
        )
        if key not in dataset.files
    }

    if missing:
        raise KeyError(f"NPZ missing arrays: {sorted(missing)}")

    images = np.asarray(dataset[image_key])

    labels = np.asarray(dataset[label_key]).reshape(-1)

    if len(images) != len(labels):
        raise ValueError(f"{split}: {len(images)} images but " f"{len(labels)} labels")

    return (
        images,
        labels.astype(
            np.int64,
            copy=False,
        ),
    )


def flatten_images(
    images: np.ndarray,
) -> np.ndarray:
    """
    Flatten N grayscale images into an N x features matrix.
    """

    images = np.asarray(images)

    if images.ndim not in {
        3,
        4,
    }:
        raise ValueError(
            "Expected image batch with shape "
            "(N, H, W) or (N, H, W, 1), "
            f"got {images.shape}"
        )

    if images.ndim == 4 and images.shape[-1] != 1:
        raise ValueError("Expected grayscale images, " f"got shape {images.shape}")

    return images.reshape(
        len(images),
        -1,
    ).astype(
        np.float32,
        copy=False,
    )
