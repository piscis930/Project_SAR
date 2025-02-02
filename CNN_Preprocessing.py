import numpy as np
import os
import rasterio
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows
from skimage.transform import resize
from collections import defaultdict
from typing import List, Dict, Any
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box


def align_and_crop_rasters(sar_dir, dem_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(dem_path) as dem:
        dem_crs = dem.crs
        dem_transform = dem.transform
        dem_bounds = dem.bounds
        dem_geometry = box(*dem_bounds)

    for filename in os.listdir(sar_dir):
        if filename.endswith(".tif"):
            sar_path = os.path.join(sar_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")

            with rasterio.open(sar_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dem_crs, src.width, src.height, *src.bounds
                )

                kwargs = src.meta.copy()
                kwargs.update(
                    {
                        "crs": dem_crs,
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )

                with rasterio.open(output_path, "w", **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dem_crs,
                        resampling=Resampling.nearest,
                    )

            # Open the reprojected file for reading and apply the mask
            with rasterio.open(output_path) as src:
                out_image, out_transform = mask(
                    src, [dem_geometry.__geo_interface__], crop=True
                )
                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

            # Write the final cropped and aligned raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

    print("Processing completed successfully!")


def align_and_expand_avalanche_rasters(avalanche_dir, dem_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(dem_path) as dem:
        dem_transform = dem.transform
        dem_crs = dem.crs
        dem_width = dem.width
        dem_height = dem.height

    for filename in os.listdir(avalanche_dir):
        if filename.endswith(".tif"):
            avalanche_path = os.path.join(avalanche_dir, filename)
            output_path = os.path.join(output_dir, f"aligned_{filename}")

            with rasterio.open(avalanche_path) as src:
                print(f"Processing: {filename}")
                print(f"Original shape: {src.shape}")
                print(f"Original data type: {src.dtypes[0]}")
                print(f"Original unique values: {np.unique(src.read(1))}")

                new_data = np.zeros((dem_height, dem_width), dtype=src.dtypes[0])

                reproject(
                    source=rasterio.band(src, 1),
                    destination=new_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dem_transform,
                    dst_crs=dem_crs,
                    resampling=Resampling.nearest,
                )

                # Special handling for specific rasters
                if filename == "ras_20240110_KB.tif":
                    new_data = np.where(new_data == 15, 0, new_data)
                elif filename == "ras_20240310_S.tif":
                    new_data = np.where(new_data == 3, 0, new_data)
                elif filename == "ras_20240322_KA.tif":
                    new_data = np.where(
                        new_data == 15, 0, new_data
                    )  # Set non-avalanche pixels (15) to 0
                    # No need to change 0 to 1, as 0 already represents avalanches in this case
                else:
                    new_data = np.where(new_data > 0, new_data, 0)

                # Count non-avalanche pixels (zeros)
                zero_count = np.count_nonzero(new_data == 0)
                total_pixels = new_data.size
                zero_percentage = (zero_count / total_pixels) * 100

                print(f"New shape: {new_data.shape}")
                print(f"New data type: {new_data.dtype}")
                print(f"New unique values: {np.unique(new_data)}")
                print(f"Non-avalanche pixels: {zero_count}")
                print(f"Percentage of non-avalanche pixels: {zero_percentage:.2f}%\n")

            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=dem_height,
                width=dem_width,
                count=1,
                dtype=src.dtypes[0],
                crs=dem_crs,
                transform=dem_transform,
            ) as dst:
                dst.write(new_data, 1)

            print(f"Processed: {filename}\n")

    print("All avalanche rasters have been processed.")


def clean_dem(dem_data):
    mask = dem_data < 1e38
    cleaned_dem = np.ma.masked_array(dem_data, ~mask)

    valid_min = np.min(cleaned_dem)
    valid_max = np.max(cleaned_dem)

    print(f"Cleaned DEM min: {valid_min}, max: {valid_max}")
    return cleaned_dem


def load_dem(dem_file):
    print("Attempting to load DEM")

    print(f"Looking for DEM file: {dem_file}")

    if os.path.exists(dem_file):
        try:
            with rasterio.open(dem_file) as src:
                dem_data = src.read(1)
                dem_data = clean_dem(dem_data)
            print(f"DEM data loaded and cleaned, shape: {dem_data.shape}")
            return dem_data
        except Exception as e:
            print(f"Error loading or cleaning the DEM data: {e}")
    else:
        print(f"DEM file not found: {dem_file}")

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
        sar_file = f"processed_SAR_{date}.tif"
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
