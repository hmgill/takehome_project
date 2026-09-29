from __future__ import annotations

import numpy as np

from project_config import CONFIG


def create_training_augmentation():
    """
    Create a deliberately mild grayscale augmentation policy.

    Augmentation is optional and should be applied only to the training
    split. The default project configuration leaves it disabled.

    The policy avoids flips, aggressive crops, and large rotations.
    """

    try:
        import albumentations as A

    except ImportError as exc:
        raise RuntimeError(
            "Albumentations is required for augmentation experiments. "
            "Install the experimental requirements first."
        ) from exc

    settings = CONFIG.augmentation

    return A.Compose(
        [
            A.Affine(
                scale=(
                    1.0 - settings.scale_limit_fraction,
                    1.0 + settings.scale_limit_fraction,
                ),
                translate_percent=(
                    -settings.translate_limit_fraction,
                    settings.translate_limit_fraction,
                ),
                rotate=(
                    -settings.rotate_limit_degrees,
                    settings.rotate_limit_degrees,
                ),
                shear=(
                    0.0,
                    0.0,
                ),
                p=settings.probability,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=settings.brightness_limit,
                contrast_limit=settings.contrast_limit,
                p=settings.probability,
            ),
        ],
        seed=CONFIG.project.random_seed,
    )


def augment_training_split(
    images: np.ndarray,
    labels: np.ndarray,
    copies_per_image: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Append augmented copies of the training split.

    The original images are retained. Validation and test data should
    never be passed to this function.
    """

    settings = CONFIG.augmentation

    copies = settings.copies_per_image if copies_per_image is None else copies_per_image

    if copies < 1:
        raise ValueError("copies_per_image must be at least 1")

    transform = create_training_augmentation()

    augmented_images = [np.asarray(images)]

    augmented_labels = [np.asarray(labels).reshape(-1)]

    for _ in range(copies):
        transformed = [transform(image=image)["image"] for image in images]

        augmented_images.append(
            np.stack(
                transformed,
                axis=0,
            )
        )

        augmented_labels.append(np.asarray(labels).reshape(-1))

    return (
        np.concatenate(
            augmented_images,
            axis=0,
        ),
        np.concatenate(
            augmented_labels,
            axis=0,
        ),
    )
