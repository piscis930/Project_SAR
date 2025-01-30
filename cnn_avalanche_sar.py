import rasterio
import re
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, TimeDistributed, LSTM
from CNN_Preprocessing import *
#from rasterio.enums import Resampling


all_bands_file_path = 'Data/all_bands/Dec23-Jan24_Cal_TC_dB_Stack_Spk.tif'

def extract_date(filename):
    # Extract date from filename (assuming format ras_YYYYMMDD)
    match = re.search(r'ras_(\d{8})', filename)
    if match:
        return match.group(1)
    return None

def load_sar_data(sar_file):
    with rasterio.open(sar_file) as src:
        return src.read()
    




#sar_float = sar.astype(np.float32)
#sar_normalized = (sar_float - sar_float.min()) / (sar_float.max() - sar_float.min())
#sar_db = 10 * np.log10(sar_normalized + 1e-10)




# Combine VV and VH channels
#X = np.stack((sar_vv_chips, sar_vh_chips), axis=-1)
#y = avalanche_chips

# Reshape for model input
#X = X.reshape(-1, chip_size, chip_size, 2)
#y = y.reshape(-1, chip_size, chip_size)

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Augment training data
X_train, y_train = augment_data(X_train, y_train)

# Now X_train, X_val, X_test are ready for model training
# y_train, y_val, y_test are the corresponding labels

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)


# NN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 2)), # 2 channels (e.g., VV, VH)
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
