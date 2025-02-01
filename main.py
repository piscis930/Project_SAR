import sys

sys.path.append("/Users/egil/Documents/avancerad_fjarranalys/SAR_Avalanche_project")

import os
from SAR_augmentation import augment_sar_avalanche_and_dem
from CNN_Preprocessing import (
    pair_avalanche_with_sar_and_dem,
    crop_sar,
    create_binary_mask,
)  # process_paired_data


sar_dir = "Data/SAR_correct"
avalanche_dir = "Data/Avalanches_corrected"
dem_dir = "Data/DEM"

# Pair SAR, Avalanche and DEM data
paired_data = pair_avalanche_with_sar_and_dem(avalanche_dir, sar_dir, dem_dir)

cropped_paired_data = crop_sar(paired_data)

# Normalize sar- and dem-data
# normalized_data = process_paired_data(cropped_paired_data)


def normalize_sar_global(data: np.ndarray) -> np.ndarray:
    data_float = data.astype(np.float32)
    data_shifted = data_float - np.min(data_float) + 1
    log_data = np.log10(data_shifted)

    p_low, p_high = np.percentile(log_data, [2, 98])
    epsilon = 1e-8
    normalized = np.clip((log_data - p_low) / (p_high - p_low + epsilon), 0, 1)

    return normalized


def normalize_dem_global(data: np.ndarray) -> np.ndarray:
    data_float = data.astype(np.float32)
    min_val, max_val = np.min(data_float), np.max(data_float)
    epsilon = 1e-8
    normalized = (data_float - min_val) / (max_val - min_val + epsilon)
    return normalized


def process_in_batches(
    data: List[Dict[str, Any]], batch_size: int = 10
) -> List[Dict[str, Any]]:
    processed_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        for item in batch:
            item["sar_data"] = normalize_sar_global(item["sar_data"])
            item["dem_data"] = normalize_dem_global(item["dem_data"])
        processed_data.extend(batch)
    return processed_data


def process_paired_data(paired_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return process_in_batches(paired_data, batch_size=10)


normalized_data = process_paired_data(cropped_paired_data)

import matplotlib.pyplot as plt
import numpy as np


def plot_sar_data(data, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.show()

    print(f"Min: {np.min(data)}, Max: {np.max(data)}")
    print(f"Mean: {np.mean(data)}, Std: {np.std(data)}")


# Plot original SAR data
plot_sar_data(cropped_paired_data[5]["sar_data"], "Original SAR Data")


def check_normalized_sar(original, normalized):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(normalized, cmap="gray")
    ax1.set_title("Normalized SAR Image")

    ax2.hist(normalized.flatten(), bins=50, range=(0, 1))
    ax2.set_title("Histogram of Normalized Values")

    plt.show()

    print(f"Normalized data - Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
    print(
        f"Normalized data - Mean: {np.mean(normalized):.4f}, Std: {np.std(normalized):.4f}"
    )

    # Check correlation with original data
    correlation = np.corrcoef(original.flatten(), normalized.flatten())[0, 1]
    print(f"Correlation with original data: {correlation:.4f}")


# Use this function on your normalized SAR data
check_normalized_sar(cropped_paired_data[5]["sar_data"], normalized_data[5]["sar_data"])


import matplotlib.pyplot as plt


def plot_hist(data, title):
    plt.hist(data.flatten(), bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")


for idx in range(3):  # First 3 items
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plot_hist(cropped_paired_data[idx]["sar_data"], f"SAR Distribution (Item {idx})")

    plt.subplot(1, 2, 2)
    plot_hist(cropped_paired_data[idx]["dem_data"], f"DEM Distribution (Item {idx})")

    plt.tight_layout()
    plt.show()


test_idx = 6  # First item
print("Original SAR sample values:")
print(
    cropped_paired_data[test_idx]["sar_data"][100:103, 100:103]
)  # Keep original data for comparison
print("\nNormalized SAR sample values:")
print(cropped_paired_data[test_idx]["sar_data"][100:103, 100:103])


import tifffile
import numpy as np

# Assuming your data is in a numpy array called 'data'
tifffile.imwrite("output.tif", cropped_paired_data[5]["sar_data"], compression="zlib")

# Create a binary mask (1=avalanch, 0=not avalanch) of avalanch data
processed_data = create_binary_mask(paired_data)

# Some testing
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

    # Clear memory after processing each pair
    del sar_chips, dem_chips, avalanche_chips

print(f"Total number of chipped pairs: {len(chipped_pairs)}")

# Some print testing
sar_chips = create_chips_new(paired_data[0]["sar_data"])
print(f"SAR chips shape: {sar_chips.shape}")
print(f"SAR chips min: {sar_chips.min()}, max: {sar_chips.max()}")
print(f"First chip sample (5x5):\n{sar_chips[0, :5, :5]}")


sar_chips = create_chips_new(paired_data[0]["sar_data"])
print(f"SAR chips shape: {sar_chips.shape}")
print(f"SAR chips min: {sar_chips.min()}, max: {sar_chips.max()}")
print(f"First chip sample (5x5):\n{sar_chips[0, :5, :5]}")


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


# To do: augment chips
augmented_chipped_pairs = augmentation(chipped_pairs)


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


# To do: Reshape and split data into Xy, train, val, test
