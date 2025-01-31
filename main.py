import sys

sys.path.append("/Users/egil/Documents/avancerad_fjarranalys/SAR_Avalanche_project")

import os
from SAR_augmentation import augment_sar_avalanche_and_dem
from CNN_Preprocessing import (pair_avalanche_with_sar_and_dem, downsample_sar, process_paired_data, create_binary_mask)
from SAR_augmentation import augmentation


sar_dir = "Data/SAR_correct"
avalanche_dir = "Data/Avalanches_corrected"
dem_dir = "Data/DEM"

# Pair SAR, Avalanche and DEM data
paired_data = pair_avalanche_with_sar_and_dem(avalanche_dir, sar_dir, dem_dir)


# Downsample sar-data to extent of dem
downsampled_paired_data = downsample_sar(paired_data)

# Normalize sar- and dem-data
normalized_data = process_paired_data(downsampled_paired_data)

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