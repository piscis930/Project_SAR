import numpy as np
import os
import rasterio
from datetime import datetime
from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows
from rasterio.merge import merge
from collections import defaultdict
import re
import pickle


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


"""
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
"""


def save_paired_data(paired_data, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(paired_data, f)
    print(f"Paired data saved to {output_file}")


def load_paired_data(input_file):
    if os.path.exists(input_file):
        with open(input_file, "rb") as f:
            paired_data = pickle.load(f)
        print(f"Paired data loaded from {input_file}")
        return paired_data
    return None


def pair_avalanche_with_sar_and_dem(avalanche_dir, sar_dir, dem_dir, output_file=None):
    if output_file and os.path.exists(output_file):
        return load_paired_data(output_file)

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
    if output_file:
        save_paired_data(paired_data, output_file)
    return paired_data


def normalize_sar(data):
    log_data = np.log10(data.astype(np.float16) + 1)
    return ((log_data - log_data.min()) / (log_data.max() - log_data.min())).astype(
        np.float16
    )


def normalize_dem(data):
    return (data - data.min()) / (data.max() - data.min())


# Split data into train, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=13
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=13
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
