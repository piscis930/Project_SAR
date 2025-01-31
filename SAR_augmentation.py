import numpy as np
import random
from skimage import exposure
from scipy.stats import gamma
from scipy.ndimage import zoom, rotate
from skimage.util import random_noise


import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
from albumentations.augmentations.geometric.transforms import (
    HorizontalFlip,
    VerticalFlip,
)
from albumentations.augmentations.geometric.rotate import Rotate
 

# To do: Clean up / remove augmentation functions



def augmentation(paired_data):
    augmentation_pipeline = A.Compose(
        [
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.VerticalFlip(p=0.5),  # Random vertical flip
            A.Rotate(limit=30, p=0.5),  # Random rotation within ±30 degrees
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            A.RandomBrightnessContrast(
                p=0.2
            ),  # Adjust brightness/contrast for SAR/DEM only
        ],
        additional_targets={"dem_data": "image", "avalanche_data": "mask"},
    )

    augmented_pairs = []
    for pair in paired_data:
        sar_chip = pair["sar_data"]
        dem_chip = pair["dem_data"]
        avalanche_chip = pair["avalanche_data"]

        # Apply augmentations
        augmented = augmentation_pipeline(
            image=sar_chip,
            dem_data=dem_chip,
            avalanche_data=avalanche_chip,
        )

        # Store augmented data
        augmented_pairs.append(
            {
                "sar_data": augmented["image"],
                "dem_data": augmented["dem_data"],
                "avalanche_data": augmented["avalanche_data"],
                "date": pair["date"],  # Keep metadata unchanged
            }
        )

    return augmented_pairs



# Define a pipeline for augmenting both images and masks
def augmentation(paired_data):
    augmentation_pipeline = A.Compose(
        [
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.VerticalFlip(p=0.5),  # Random vertical flip
            A.Rotate(limit=30, p=0.5),  # Random rotation within ±30 degrees
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            A.RandomBrightnessContrast(
                p=0.2
            ),  # Adjust brightness/contrast for SAR/DEM only
        ],
        additional_targets={"dem_data": "image", "avalanche_data": "mask"},
    )

    augmented_pairs = []

    for pair in paired_data:
        sar = pair["sar_data"]
        dem = pair["dem_data"]
        avalanche_mask = pair["avalanche_data"]

        # Apply augmentation
        augmented = augmentation_pipeline(
            image=sar, dem_data=dem, avalanche_data=avalanche_mask
        )

        # Extract augmented data
        augmented_sar = augmented["image"]
        augmented_dem = augmented["dem_data"]
        augmented_mask = augmented["avalanche_data"]

        # Save the augmented pair back into a new list
        augmented_pairs.append(
            {
                "sar_data": augmented_sar,
                "dem_data": augmented_dem,
                "avalanche_data": augmented_mask,
                "date": pair["date"],
            }
        )

    return augmented_pairs


def augment_sar_avalanche_and_dem(sar_data, avalanche_data, dem_data):
    # Apply spatial augmentations to all three
    augmented_sar = sar_data.copy()
    augmented_avalanche = avalanche_data.copy()
    augmented_dem = dem_data.copy()

    if random.random() < 0.3:
        angle = random.choice([90, 180, 270])
        augmented_sar = rotate(augmented_sar, angle, axes=(1, 2), reshape=False)
        augmented_avalanche = rotate(augmented_avalanche, angle, reshape=False)
        augmented_dem = rotate(augmented_dem, angle, reshape=False)

    if random.random() < 0.3:
        scale_factor = random.uniform(0.9, 1.1)
        augmented_sar = zoom(augmented_sar, (1, scale_factor, scale_factor), order=1)
        augmented_avalanche = zoom(
            augmented_avalanche, scale_factor, order=0
        )  # Nearest neighbor for labels
        augmented_dem = zoom(
            augmented_dem, scale_factor, order=1
        )  # Linear interpolation for DEM

    # Intensity-based transformations (apply only to SAR)
    if random.random() < 0.5:
        augmented_sar = add_speckle_noise(augmented_sar)
    if random.random() < 0.5:
        augmented_sar = adjust_contrast(augmented_sar)
    if random.random() < 0.4:
        augmented_sar = gamma_histogram_specification(augmented_sar)

    return augmented_sar, augmented_avalanche, augmented_dem


# Add speckle noise
def add_speckle_noise(image, intensity=0.1):
    noise = np.random.normal(0, intensity, image.shape)
    return image + image * noise


# Change Contrast
def adjust_contrast(image, factor=1.5):
    return exposure.adjust_gamma(image, factor)


# Alpha blend
def alpha_blend(foreground, background, alpha=0.7):
    return cv2.addWeighted(foreground, alpha, background, 1 - alpha, 0)


# Rotate
def rotate_image(image, angle=10):
    return rotate(image, angle, reshape=False)


# Scale
def scale_image(image, scale_factor=1.1):
    return zoom(image, scale_factor)


# Gamma Distribution-based Histogram
def gamma_histogram_specification(image, a=2, scale=2):
    hist, bin_edges = np.histogram(image.flatten(), bins=256, density=True)
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    gamma_dist = gamma.rvs(a, scale=scale, size=256)
    gamma_cdf = np.cumsum(gamma_dist)
    gamma_cdf = 255 * gamma_cdf / gamma_cdf[-1]

    interp_values = np.interp(cdf, gamma_cdf, range(256))
    return interp_values[image].astype(np.uint8)


def augment_sar_image(image):
    augmented = image.copy()

    # Randomly apply augmentations
    if random.random() < 0.5:
        augmented = add_speckle_noise(augmented)
    if random.random() < 0.5:
        augmented = adjust_contrast(augmented)
    if random.random() < 0.3:
        augmented = rotate_image(augmented)
    if random.random() < 0.3:
        augmented = scale_image(augmented)
    if random.random() < 0.4:
        augmented = gamma_histogram_specification(augmented)

    return augmented
