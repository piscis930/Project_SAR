import sys

sys.path.append("/Users/egil/Documents/avancerad_fjarranalys/SAR_Avalanche_project")

import os
from SAR_augmentation import augment_sar_avalanche_and_dem
from CNN_Preprocessing import (
    normalize_dem,
    normalize_sar,
    pair_avalanche_with_sar_and_dem,
)


sar_dir = "Data/SAR_correct"
avalanche_dir = "Data/Avalanches_corrected"
dem_dir = "Data/DEM"


# Pair SAR, Avalanche and DEM data
paired_data = pair_avalanche_with_sar_and_dem(avalanche_dir, sar_dir, dem_dir)


import numpy as np
from typing import List, Dict, Any


def normalize_sar_in_chunks(data: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    h, w = data.shape
    normalized = np.zeros_like(data, dtype=np.float32)
    epsilon = 1e-8  # Add this line at the beginning of the function
    for i in range(0, h, chunk_size):
        chunk = data[i : i + chunk_size, :]
        print(f"Chunk min: {chunk.min()}, max: {chunk.max()}")

        chunk = chunk.astype(np.float32)  # Ensure float32 type
        chunk_shifted = chunk - chunk.min() + 1  # Shift to positive values
        log_chunk = np.log10(chunk_shifted)
        print(f"Log chunk min: {log_chunk.min()}, max: {log_chunk.max()}")
        min_val, max_val = log_chunk.min(), log_chunk.max()
        if min_val == max_val:
            normalized[i : i + chunk_size, :] = 0
        else:
            # Replace this line
            normalized[i : i + chunk_size, :] = (log_chunk - min_val) / (
                max_val - min_val + epsilon
            )
        print(
            f"Normalized chunk min: {normalized[i:i+chunk_size, :].min()}, max: {normalized[i:i+chunk_size, :].max()}"
        )
    return normalized


def normalize_dem_in_chunks(data: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    h, w = data.shape
    normalized = np.zeros_like(data, dtype=np.float32)
    for i in range(0, h, chunk_size):
        chunk = data[i : i + chunk_size, :]
        min_val, max_val = chunk.min(), chunk.max()
        normalized[i : i + chunk_size, :] = (chunk - min_val) / (max_val - min_val)
    return normalized


def process_in_batches(
    data: List[Dict[str, Any]], batch_size: int = 10
) -> List[Dict[str, Any]]:
    processed_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        for item in batch:
            item["sar_data"] = normalize_sar_in_chunks(item["sar_data"])
            item["dem_data"] = normalize_dem_in_chunks(item["dem_data"])
        processed_data.extend(batch)
    return processed_data


# Main processing function
def process_paired_data(paired_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return process_in_batches(paired_data, batch_size=10)


processed_data = process_paired_data(paired_data)


print(f"Processed {len(processed_data)} items")
print(processed_data[0]["sar_data"][:10, :10])  # Print first 10x10 elements


for pair in paired_data:
    avalanche_data = pair["avalanche_data"]
    binary_mask = avalanche_data > 0
    pair["binary_avalanche_data"] = binary_mask.astype(int)

for pair in paired_data:
    pair["avalanche_data"] = pair["binary_avalanche_data"]
    del pair["binary_avalanche_data"]

for i, pair in enumerate(paired_data):
    print(f"Paired Data {i + 1}:")
    date = pair["date"]
    print(f"Date: {date}")
    avalanche_data = pair["avalanche_data"]
    SAR_data = pair["sar_data"]
    print(f"Avalanche Data Shape: {avalanche_data.shape}")
    print(f"Min Value: {avalanche_data.min()}, Max Value: {avalanche_data.max()}")
    print(f"Mean Value: {avalanche_data.mean()}")
    print(f"SAR Data Shape: {SAR_data.shape}")
    print(f"Min Value: {SAR_data.min()}, Max Value: {SAR_data.max()}")
    print(f"Mean Value: {SAR_data.mean()}")
    print("-" * 30)


"""
chipped_pairs = []
chip_size = 128
for pair in paired_data:
    sar_data = pair['sar_data']
    dem_data = pair['dem_data']
    avalanche_data = pair['avalanche_data']
    date = pair['date']

    # Create chips for each data type
    sar_chips = create_chips(sar_data)
    dem_chips = create_chips(dem_data)
    avalanche_chips = create_chips(avalanche_data)

    # Reshape the chips to be 3D arrays
    sar_chips = sar_chips.reshape(-1, chip_size, chip_size)
    dem_chips = dem_chips.reshape(-1, chip_size, chip_size)
    avalanche_chips = avalanche_chips.reshape(-1, chip_size, chip_size)

    # Create new pairs for each set of chips
    for sar_chip, dem_chip, avalanche_chip in zip(sar_chips, dem_chips, avalanche_chips):
        chipped_pairs.append({
            'sar_data': sar_chip,
            'dem_data': dem_chip,
            'avalanche_data': avalanche_chip,
            'date': date
        })

# Now chipped_pairs contains all the chipped data
print(f"Total number of chipped pairs: {len(chipped_pairs)}")
"""


# Todo Fix chipping
import numpy as np


def create_chips_new(data, chip_size=128, max_chips=1000):
    h, w = data.shape
    chips = []
    for i in range(0, h - chip_size + 1, chip_size):
        for j in range(0, w - chip_size + 1, chip_size):
            chip = np.copy(data[i : i + chip_size, j : j + chip_size])
            chips.append(chip)
            if len(chips) >= max_chips:
                return np.array(chips)
    return np.array(chips)


chipped_pairs = []

for pair in paired_data:
    sar_data = pair["sar_data"]
    dem_data = pair["dem_data"]
    avalanche_data = pair["avalanche_data"]
    date = pair["date"]

    # Create chips for each data type
    sar_chips = create_chips_new(sar_data)
    dem_chips = create_chips_new(dem_data)
    avalanche_chips = create_chips_new(avalanche_data)

    # Create new pairs for each set of chips
    for sar_chip, dem_chip, avalanche_chip in zip(
        sar_chips, dem_chips, avalanche_chips
    ):
        chipped_pairs.append(
            {
                "sar_data": sar_chip,
                "dem_data": dem_chip,
                "avalanche_data": avalanche_chip,
                "date": date,
            }
        )

    # Optional: Clear memory after processing each pair
    del sar_chips, dem_chips, avalanche_chips

print(f"Total number of chipped pairs: {len(chipped_pairs)}")


sar_chips = create_chips_new(paired_data[0]["sar_data"])
print(f"SAR chips shape: {sar_chips.shape}")
print(f"SAR chips min: {sar_chips.min()}, max: {sar_chips.max()}")
print(f"First chip sample (5x5):\n{sar_chips[0, :5, :5]}")


from skimage.util import view_as_windows


def create_chips(data, chip_size=128):
    return view_as_windows(data, (chip_size, chip_size), step=chip_size)


def create_chips_generator(data, chip_size=128):
    h, w = data.shape
    print(f"Input data shape: {data.shape}, dtype: {data.dtype}")
    print(f"Input data range: {data.min()} to {data.max()}")
    for i in range(0, h - chip_size + 1, chip_size):
        for j in range(0, w - chip_size + 1, chip_size):
            chip = np.array(data[i : i + chip_size, j : j + chip_size], copy=True)
            print(
                f"Chip at ({i},{j}): shape {chip.shape}, dtype {chip.dtype}, range {chip.min()} to {chip.max()}"
            )
            yield chip


first_128x128_before = paired_data[0]["sar_data"][:128, :128]
print(first_128x128_before)
sar_chips = create_chips_generator(first_128x128_before, chip_size=128)
first_chip_after = next(sar_chips)
print(f"Shape before chipping: {first_128x128_before.shape}")
print(f"Shape after chipping: {first_chip_after.shape}")
print(
    f"Range before chipping: {first_128x128_before.min()} to {first_128x128_before.max()}"
)
print(f"Range after chipping: {first_chip_after.min()} to {first_chip_after.max()}")


import albumentations as A


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


# Usage Example
augmented_chipped_pairs = augmentation(chipped_pairs)


from SAR_augmentation import augmentation


augmented_pairs = augmentation(paired_data)


augmented_patches = []
for pair in paired_data:
    sar_data = pair["sar_data"]
    avalanche_data = pair["avalanche_data"]
    dem_data = pair["dem_data"]

    # Augment full images
    augmented_sar, augmented_avalanche, augmented_dem = augment_sar_avalanche_and_dem(
        sar_data, avalanche_data, dem_data
    )

    # Create patches
    sar_patches = create_chips(augmented_sar, chip_size=128)
    avalanche_patches = create_chips(augmented_avalanche, chip_size=128)
    dem_patches = create_chips(augmented_dem, chip_size=128)

    # Flatten patches into a list
    for i in range(sar_patches.shape[0]):
        for j in range(sar_patches.shape[1]):
            augmented_patches.append(
                (sar_patches[i, j], avalanche_patches[i, j], dem_patches[i, j])
            )
