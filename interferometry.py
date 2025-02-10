from snappy import ProductIO, HashMap, GPF
import os



def ALOS_deskewing(parameters, products):
    deskewed_products = []
    for product in products:
        deskewed_product = GPF.createProduct('ALOS-Deskewing', parameters, product)
        deskewed_products.append(deskewed_product)
        return deskewed_products

deskewing_parameters = HashMap()
deskewing_parameters.put('applyRadiometricCorrection', False)

file_dir = 'path'
products = []
for filename in os.listdir(file_dir):
        if filename.endswith(".safe"):
            file_path = os.path.join(file_dir, filename)
            product = ProductIO.readProduct(file_path)
            products.append(product)

deskewed_products = ALOS_deskewing(deskewing_parameters, products)




def co_registration(master_file, slave_file, paramterers):

    coregister = GPF.createOperator('SAR-Coregistration', paramterers)
    sourceProducts = HashMap()
    sourceProducts.put('Master', master_file)
    sourceProducts.put('Slave', slave_file)

    return coregister.execute(sourceProducts)


co_registration_parameters = HashMap()
co_registration_parameters.put('demName', 'SRTM 1Sec HGT')
co_registration_parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
co_registration_parameters.put('resamplingType', 'BILINEAR_INTERPOLATION')
co_registration_parameters.put('maskOutAreaWithoutElevation', True)
co_registration_parameters.put('outputRangeAzimuthOffset', True)
co_registration_parameters.put('outputDerampDemodPhase', True)

co_registrated_product = co_registration(deskewed_products[0], deskewed_products[1], co_registration_parameters)





# Load the Sentinel-1 product

product = ProductIO.readProduct('path/to/Sentinel1_SAFE_directory')

# Apply orbit file
parameters = HashMap()
parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
parameters.put('polyDegree', 3)
product_orbit = GPF.createProduct('Apply-Orbit-File', parameters, product)

# Thermal noise removal
parameters = HashMap()
product_tnr = GPF.createProduct('ThermalNoiseRemoval', parameters, product_orbit)

# Calibration
parameters = HashMap()
product_cal = GPF.createProduct('Calibration', parameters, product_tnr)

# Speckle filtering
parameters = HashMap()
parameters.put('filter', 'Lee')
product_sf = GPF.createProduct('Speckle-Filter', parameters, product_cal)

# Terrain correction
parameters = HashMap()
parameters.put('demName', 'SRTM 3Sec')
product_tc = GPF.createProduct('Terrain-Correction', parameters, product_sf)

# Save the final product
ProductIO.writeProduct(product_tc, 'path/to/output_product.dim', 'BEAM-DIMAP')




    