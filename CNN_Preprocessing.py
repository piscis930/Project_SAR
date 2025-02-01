import numpy as np
import os
import rasterio
import re
import copy
from datetime import datetime
from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows
from skimage.transform import resize
from collections import defaultdict
from typing import List, Dict, Any


def load_dem(dem_dir):
    print("Attempting to load DEM")
    dem_file = "Comb_10m_DTM.tif"
    dem_path = os.path.join(dem_dir, dem_file)
    print(f"Looking for DEM file: {dem_path}")

    if os.path.exists(dem_path):
        try:
            with rasterio.open(dem_path) as src:
                dem_data = src.read(1)
            print(f"DEM data loaded, shape: {dem_data.shape}")
            return dem_data
        except Exception as e:
            print(f"Error loading DEM data: {e}")
    else:
        print(f"DEM file not found: {dem_path}")

    return None


def load_sar_data(sar_path):
    try:
        with rasterio.open(sar_path) as src:
            return src.read(1)
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening {sar_path}: {e}")
    except Exception as e:
        print(f"Unexpected error loading {sar_path}: {e}")
    return None


def load_avalanche_data(avalanche_path):
    try:
        with rasterio.open(avalanche_path) as src:
            # Read the first band (assuming single-band SAR data)

            return src.read(1)

    except Exception as e:
        print(f"Error loading avalanche data from {avalanche_path}: {e}")
        return None


def extract_date_from_filename(filename):
    pattern = r"ras_(\d{8})(?:_[A-Z]+)?(?:_J)?\.tif"
    match = re.search(pattern, filename)
    return match.group(1) if match else None


def group_avalanche_files_by_date(avalanche_dir):
    files_by_date = defaultdict(list)
    for filename in os.listdir(avalanche_dir):
        if filename.endswith(".tif"):
            date = extract_date_from_filename(filename)
            if date:
                files_by_date[date].append(filename)
    return files_by_date


def combine_avalanche_data(avalanche_data_list):
    return np.maximum.reduce(avalanche_data_list)


def pair_avalanche_with_sar_and_dem(avalanche_dir, sar_dir, dem_dir):
    print("Starting pair_avalanche_with_sar_and_dem function")
    paired_data = []
    grouped_files = group_avalanche_files_by_date(avalanche_dir)
    print(f"Grouped files: {grouped_files}")

    # Load DEM once
    dem_data = load_dem(dem_dir)
    if dem_data is None:
        print("Failed to load DEM data. Exiting function.")
        return paired_data

    for date, avalanche_files in grouped_files.items():
        print(f"Processing date: {date}")
        sar_file = f"SAR_{date}.tif"
        sar_path = os.path.join(sar_dir, sar_file)
        print(f"Looking for SAR file: {sar_path}")

        if os.path.exists(sar_path):
            print(f"SAR file found: {sar_path}")
            try:
                sar_data = load_sar_data(sar_path)
                print(f"SAR data loaded, shape: {sar_data.shape}")
                print("SAR data type:", sar_data.dtype)
                print("SAR data contains NaN:", np.isnan(sar_data).any())
                print("SAR data min:", np.nanmin(sar_data))
                print("SAR data max:", np.nanmax(sar_data))
            except Exception as e:
                print(f"Error loading SAR data: {e}")
                continue

            avalanche_data_list = []
            for avalanche_file in avalanche_files:
                avalanche_path = os.path.join(avalanche_dir, avalanche_file)
                print(f"Loading avalanche file: {avalanche_path}")
                try:
                    with rasterio.open(avalanche_path) as src:
                        avalanche_data = src.read(1)
                    print(f"Avalanche data loaded, shape: {avalanche_data.shape}")
                    avalanche_data_list.append(avalanche_data)
                except Exception as e:
                    print(f"Error loading avalanche data: {e}")

            if avalanche_data_list:
                combined_avalanche_data = combine_avalanche_data(avalanche_data_list)
                print(
                    f"Combined avalanche data, shape: {combined_avalanche_data.shape}"
                )

                paired_data.append(
                    {
                        "date": datetime.strptime(date, "%Y%m%d").date(),
                        "avalanche_data": combined_avalanche_data,
                        "sar_data": sar_data,
                        "dem_data": dem_data,
                        "avalanche_files": avalanche_files,
                        "sar_file": sar_file,
                    }
                )
                print(f"Data paired for date: {date}")
            else:
                print(f"No valid avalanche data for date: {date}")
        else:
            print(f"SAR file not found: {sar_path}")

    print(f"Total paired data: {len(paired_data)}")

    return paired_data


def crop_sar(paired_data):
    cropped_paired_data = []
    for item in paired_data:
        new_item = item.copy()
        sar_data = item["sar_data"]
        dem_data = item["dem_data"]

        # Find the non-zero region in the SAR data
        non_zero = np.nonzero(sar_data)
        min_row, max_row = np.min(non_zero[0]), np.max(non_zero[0])
        min_col, max_col = np.min(non_zero[1]), np.max(non_zero[1])

        # Crop the SAR data to the non-zero region
        cropped_sar = sar_data[min_row : max_row + 1, min_col : max_col + 1]

        print(f"Original SAR Dimensions: {sar_data.shape}")
        print(f"Cropped SAR Dimensions: {cropped_sar.shape}")
        print(f"DEM Dimensions: {dem_data.shape}")

        # Verify that the cropped SAR dimensions match the DEM dimensions
        if cropped_sar.shape != dem_data.shape:
            raise ValueError(
                f"Cropped SAR dimensions {cropped_sar.shape} do not match DEM dimensions {dem_data.shape}"
            )

        new_item["sar_data"] = cropped_sar
        cropped_paired_data.append(new_item)
    return cropped_paired_data


"""
def normalize_sar_in_chunks(data: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    h, w = data.shape
    normalized = np.zeros_like(data, dtype=np.float32)
    epsilon = 1e-8
    for i in range(0, h, chunk_size):
        chunk = data[i : i + chunk_size, :]
        print(f"Chunk min: {chunk.min()}, max: {chunk.max()}")

        chunk = chunk.astype(np.float32)
        chunk_shifted = chunk - chunk.min() + 1
        log_chunk = np.log10(chunk_shifted)
        print(f"Log chunk min: {log_chunk.min()}, max: {log_chunk.max()}")
        min_val, max_val = log_chunk.min(), log_chunk.max()
        if min_val == max_val:
            normalized[i : i + chunk_size, :] = 0
        else:
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


def process_paired_data(paired_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return process_in_batches(paired_data, batch_size=10)
"""


def create_binary_mask(paired_data):
    for pair in paired_data:
        avalanche_data = pair["avalanche_data"]
        binary_mask = avalanche_data > 0
        pair["binary_avalanche_data"] = binary_mask.astype(int)

    for pair in paired_data:
        pair["avalanche_data"] = pair["binary_avalanche_data"]
        del pair["binary_avalanche_data"]

    return paired_data


# To do: Fix and test chippingg funtion with the downsampled sar data


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


# Split data into train, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=13
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=13
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
