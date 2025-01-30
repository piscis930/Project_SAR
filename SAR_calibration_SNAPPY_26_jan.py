# import sys
# sys.path.insert(0, r"C:\snap-python\snappy")
import os
from snappy import HashMap
from Calibration import (
    load_sar_products,
    subset_SAR,
    thermal_noise_removal,
    apply_orbit_file,
    calibrate_SAR,
    speckle_filtering,
    terrain_correction,
)

base_dir = os.path.join(
    r"C:\Users\egila\Documents\studier_VT25\Advanced_Remote_Sensing\Uppgift_1",
    "SAR_Data_test",
)
subset_dir = os.path.join(
    r"C:\Users\egila\Documents\studier_VT25\Advanced_Remote_Sensing\Uppgift_1", "Subset"
)
thermal_noise_removed_dir = os.path.join(
    r"C:\Users\egila\Documents\studier_VT25\Advanced_Remote_Sensing\Uppgift_1", "Thermal_Noise_Removed"
)
apply_orbit_file_dir = os.path.join(
    r"C:\Users\egila\Documents\studier_VT25\Advanced_Remote_Sensing\Uppgift_1", "Applied_Orbit_File"
)
calibrated_dir = os.path.join(
    r"C:\Users\egila\Documents\studier_VT25\Advanced_Remote_Sensing\Uppgift_1",
    "Calibrated",
)
speckle_filtered_dir = os.path.join(
    r"C:\Users\egila\Documents\studier_VT25\Advanced_Remote_Sensing\Uppgift_1",
    "Speckle_Filtered",
)
terrain_corrected_dir = os.path.join(
    r"C:\Users\egila\Documents\studier_VT25\Advanced_Remote_Sensing\Uppgift_1",
    "Terrain_Corrected",
)
conversion_to_dB_dir = os.path.join(
    r"C:\Users\egila\Documents\studier_VT25\Advanced_Remote_Sensing\Uppgift_1",
    "converted_to_dB",
    )


for product in load_sar_products(base_dir):
    try:
        #Subsetting
        lat_min, lat_max = 69.17580, 69.98087
        lon_min, lon_max = 18.92285, 20.41071
        print("Starting subset...")
        subset_product_file = subset_SAR(
            product, lat_min, lat_max, lon_min, lon_max, subset_dir
        )
        print(f"Subset completed. File: {subset_product_file}")

        # Remove thermal noise
        thermal_noise_removed_prouduct_file = thermal_noise_removal(subset_product_file, )

        # Apply orbit file
        orbit_parameters = HashMap()
        orbit_parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
        orbit_parameters.put('polyDegree', 3)
        orbit_parameters.put('continueOnFail', True)
        orbit_file_applide_prouduct_file = apply_orbit_file(thermal_noise_removed_prouduct_file, orbit_parameters, apply_orbit_file_dir)

        # Calibrating
        print("Starting calibration...")
        calibrated_product_file = calibrate_SAR(subset_product_file, calibrated_dir)
        print(f"Calibration completed. File: {calibrated_product_file}")

        # Speckle filtering
        print("Starting speckle filtering...")
        speckle_filtered_file = speckle_filtering(
            calibrated_product_file, speckle_filtered_dir
        )
        print(f"Speckle filtering completed. File: {speckle_filtered_file}")

        # Terrain Correction
        print("Starting terrain correction...")
        speckle_filtering_parameters = HashMap()
        speckle_filtering_parameters.put("demName", "SRTM 1Sec HGT")
        speckle_filtering_parameters.put("mapProjection", "EPSG:32633")
        speckle_filtering_parameters.put("pixelSpacingInMeter", 10.0)
        speckle_filtering_parameters.put("sourceBands", "Sigma0_VV")
        speckle_filtering_parameters.put("demApplyElevation", True)
        speckle_filtering_parameters.put("saveSelectedSourceBand", True)
        speckle_filtering_parameters.put("nodataValueAtSea", False)

        terrain_corrected_file = terrain_correction(
            speckle_filtering_parameters, speckle_filtered_file, terrain_corrected_dir
        )
        print(f"Terrain correction completed. File: {terrain_corrected_file}")

        # Convert to dB
        
        conversion_to_dB_parameters = HashMap()
        conversion_to_dB_parameters.put('sourceBands', 'Sigma0_VH,Sigma0_VV')  

        converted_to_dB_file = terrain_correction(
            terrain_corrected_file, conversion_to_dB_parameters, conversion_to_dB_dir
        )
        print(f"Terrain correction completed. File: {converted_to_dB_file}")
        



        print(f"Successfully processed: {product.getName()}")
    except Exception as e:
        print(f"Error processing {product.getName()}: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        if product:
            product.dispose()

