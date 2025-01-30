import os
import sys

sys.path.insert(0, r"C:\snap-python\snappy")
from snappy import ProductIO, HashMap, GPF


def load_sar_products(base_directory):
    for folder in os.listdir(base_directory):
        if folder.endswith(".SAFE"):
            manifest_path = os.path.join(base_directory, folder, "manifest.safe")
            if os.path.exists(manifest_path):
                print(f"Loading product: {folder}")
                try:
                    product = ProductIO.readProduct(manifest_path)
                    yield product
                except Exception as e:
                    print(f"Error loading {folder}: {str(e)}")




def subset_SAR(product, lat_min, lat_max, lon_min, lon_max, output_dir):
    # Define the subsetting parameters
    subset_parameters = HashMap()
    subset_parameters.put(
        "geoRegion",
        f"POLYGON(({lon_min} {lat_max}, {lon_max} {lat_max}, {lon_max} {lat_min}, {lon_min} {lat_min}, {lon_min} {lat_max}))",
    )
    subset_parameters.put("copyMetadata", True)

    # Apply Subset
    print("Applying Subset...")
    subset_product = GPF.createProduct("Subset", subset_parameters, product)

    original_folder_name = os.path.basename(
        os.path.dirname(product.getFileLocation().getPath())
    )

    # Create the new file name
    new_file_name = f"{original_folder_name}_subset.dim"

    # Save the subset product to file
    output_file = os.path.join(output_dir, new_file_name)
    print(f"Saving subset product to {output_file}")
    ProductIO.writeProduct(subset_product, output_file, "BEAM-DIMAP")

    # Close the product to free up memory
    subset_product.dispose()

    return output_file


def thermal_noise_removal(product_file, parameters, thermal_noise_removed_dir):
# Thermal Noise Removal
    subset_product = ProductIO.readProduct(product_file)

    thermal_noise_removed_product = GPF.createProduct('ThermalNoiseRemoval', parameters, subset_product)

    # Generate output filename based on input file
    input_filename = os.path.basename(product_file)
    output_filename = f"{os.path.splitext(input_filename)[0]}_thermal_noise_removed.dim"
    output_file = os.path.join(thermal_noise_removed_dir, output_filename)

    print(f"Saving thermal noise removed product to {output_file}")
    ProductIO.writeProduct(thermal_noise_removed_product, output_file, "BEAM-DIMAP")

    subset_product.dispose()
    thermal_noise_removed_product.dispose()

    return output_file


def apply_orbit_file(product_file, parameters, apply_orbit_file_dir):
# Apply Orbit File
    product = ProductIO.readProduct(product_file)

    orbit_product = GPF.createProduct('Apply-Orbit-File', parameters, product)

     # Generate output filename based on input file
    input_filename = os.path.basename(product_file)
    output_filename = f"{os.path.splitext(input_filename)[0]}_orbit_file_applied.dim"
    output_file = os.path.join(apply_orbit_file_dir, output_filename)

    print(f"Saving thermal noise removed product to {output_file}")
    ProductIO.writeProduct(orbit_product, output_file, "BEAM-DIMAP")

    orbit_product.dispose()
    product.dispose()

    return output_file



def calibrate_SAR(product_file, calibrated_dir):
    print("Applying Calibration...")
    product = ProductIO.readProduct(product_file)

    calibration_parameters = HashMap()
    calibration_parameters.put("outputSigmaBand", True)
    calibration_parameters.put("selectedPolarisations", "VV")
    calibration_parameters.put("sourceBands", "Intensity_VV")

    calibrated_product = GPF.createProduct(
        "Calibration", calibration_parameters, product
    )

    # Generate output filename based on input file
    input_filename = os.path.basename(product_file)
    output_filename = f"{os.path.splitext(input_filename)[0]}_calibrated.dim"
    output_file = os.path.join(calibrated_dir, output_filename)

    print(f"Saving calibrated product to {output_file}")
    ProductIO.writeProduct(calibrated_product, output_file, "BEAM-DIMAP")

    product.dispose()
    calibrated_product.dispose()

    return output_file


def speckle_filtering(product_file, speckle_filtered_dir):
    print("Starting speckle filtering...")


    product = ProductIO.readProduct(product_file)
    if product is None:
        print(
            f"Error: Failed to read product from {product_file}"
        )
        return None

    speckle_filter_parameters = HashMap()
    speckle_filter_parameters.put("filter", "Refined Lee")  # Choose the desired filter
    speckle_filtered_product = GPF.createProduct(
        "Speckle-Filter", speckle_filter_parameters, product
    )

    # Generate output filename based on input file
    input_filename = os.path.basename(product_file)
    output_filename = f"{os.path.splitext(input_filename)[0]}_speckle_filtered.dim"
    output_file = os.path.join(speckle_filtered_dir, output_filename)

    print(f"Saving speckle filtered product to {output_file}")
    ProductIO.writeProduct(speckle_filtered_product, output_file, "BEAM-DIMAP")

    # Close the product to free up memory
    speckle_filtered_product.dispose()
    product.dispose()

    return output_file


# Apply Range-Doppler Terrain Correction
def terrain_correction(parameters, product_file, terrain_corrected_dir):

    product = ProductIO.readProduct(product_file)
    terrain_corrected_product = GPF.createProduct(
        "Terrain-Correction", parameters, product
    )

    # Generate output filename based on input file
    input_filename = os.path.basename(product_file)
    output_filename = f"{os.path.splitext(input_filename)[0]}_terrain_corrected.dim"
    output_file = os.path.join(terrain_corrected_dir, output_filename)

    print(f"Saving terrain corrected product to {output_file}")
    ProductIO.writeProduct(terrain_corrected_product, output_file, "BEAM-DIMAP")

    # Close the product to free up memory
    product.dispose()
    terrain_corrected_product.dispose()

    return output_file


def conversion_to_dB(product_file, parameters, conversion_to_dB_dir):
    product = ProductIO.readProduct(product_file)


    linear_to_db_product = GPF.createProduct('LinearToFromdB', parameters, product)

       # Generate output filename based on input file
    input_filename = os.path.basename(product_file)
    output_filename = f"{os.path.splitext(input_filename)[0]}_converted_to_dB.dim"
    output_file = os.path.join(conversion_to_dB_dir, output_filename)

    print(f"Saving terrain corrected product to {output_file}")
    ProductIO.writeProduct(linear_to_db_product, output_file, "BEAM-DIMAP")

    # Close the product to free up memory
    linear_to_db_product.dispose()
    product.dispose()

    return output_file


def co_registration():
    master = ProductIO.readProduct('path/to/master_image.zip')
    slave = ProductIO.readProduct('path/to/slave_image.zip')

    parameters = HashMap()
    parameters.put('demName', 'SRTM 1Sec HGT')
    parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('resamplingType', 'BILINEAR_INTERPOLATION')
    parameters.put('maskOutAreaWithoutElevation', True)
    parameters.put('outputRangeAzimuthOffset', True)
    parameters.put('outputDerampDemodPhase', True)


    coregister = GPF.createOperator('SAR-Coregistration', parameters)
    sourceProducts = HashMap()
    sourceProducts.put('Master', master)
    sourceProducts.put('Slave', slave)
    coregistered = coregister.execute(sourceProducts)
