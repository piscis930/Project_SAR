#import sys

#sys.path.append("/Users/egil/Documents/avancerad_fjarranalys/SAR_Avalanche_project")

import gc
import numpy as np

# from SAR_augmentation import augment_sar_avalanche_and_dem
from CNN_Preprocessing import (
    align_and_crop_rasters,
    align_and_expand_avalanche_rasters,
    pair_avalanche_with_sar_and_dem,
    process_paired_data,
    create_binary_mask,
    create_chips,
    custom_train_test_split
)


sar_dir = "Data/SAR_correct"
avalanche_dir = "Data/Avalanches_corrected"
dem_file_path = "Data/DEM/Comb_10m_DTM.tif"
aligned_and_cropped_dir_SAR = "Data/Cropped_SAR"
aligned_and_filled_directory_avalanche = "Data/Aligned_and_filled_avalanche_data"

align_and_expand_avalanche_rasters(
    avalanche_dir, dem_file_path, aligned_and_filled_directory_avalanche
)


align_and_crop_rasters(sar_dir, dem_file_path, aligned_and_cropped_dir_SAR)

# Pair SAR, Avalanche and DEM data
aligned_paired_data = pair_avalanche_with_sar_and_dem(
    aligned_and_filled_directory_avalanche, aligned_and_cropped_dir_SAR, dem_file_path
)

# Normalize SAR and DEM data
normalized_data = process_paired_data(aligned_paired_data)
# Discard old dataset to free up memory
del aligned_paired_data
gc.collect()

# Create a binary mask (1=avalanche, 0=not avalanche) of avalanche data
processed_data = create_binary_mask(normalized_data)
del normalized_data
gc.collect()


def crop_sar_data(processed_data):
    for data in processed_data:
        if data["sar_data"].shape != data["dem_data"].shape:
            data["sar_data"] = data["sar_data"][:-1, :-1]
    return processed_data


processed_data = crop_sar_data(processed_data)


chipped_pairs = []

for pair in processed_data:
    sar_data = pair["sar_data"]
    dem_data = pair["dem_data"]
    avalanche_data = pair["avalanche_data"]
    date = pair["date"]

    # Create generators for each data type
    sar_chips = create_chips(sar_data)
    dem_chips = create_chips(dem_data)
    avalanche_chips = create_chips(avalanche_data)

    # Iterate through the generators simultaneously
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

del processed_data
gc.collect()


X_sar = np.memmap("X_SARs.dat", dtype="float32", mode="w+", shape=(73008, 128, 128))
X_dem = np.memmap("X_DEMs.dat", dtype="float32", mode="w+", shape=(73008, 128, 128))
y = np.memmap("ys.dat", dtype="float32", mode="w+", shape=(73008, 128, 128))

for i, pair in enumerate(chipped_pairs):
    X_sar[i] = pair["sar_data"]
    X_dem[i] = pair["dem_data"]
    y[i] = pair["avalanche_data"]

X_sar = X_sar.reshape(-1, 128, 128, 1)
X_dem = X_dem.reshape(-1, 128, 128, 1)
y = y.reshape(-1, 128, 128, 1)



X_sar_train, X_sar_test, X_dem_train, X_dem_test, y_train, y_test = (
    custom_train_test_split(
        X_sar, X_dem, y, test_size=0.2, random_state=42, batch_size=1000
    )
)

X_sar_train, X_sar_val, X_dem_train, X_dem_val, y_train, y_val = (
    custom_train_test_split(
        X_sar_train,
        X_dem_train,
        y_train,
        test_size=0.25,
        random_state=42,
        batch_size=1000,
    )
)

# To do create dataset, create model, compile and train model, evaluate


import tensorflow as tf

print(tf.__version__)


def create_dataset(X_sar, X_dem, y, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(((X_sar, X_dem), y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


train_dataset = create_dataset(X_sar_train, X_dem_train, y_train)
val_dataset = create_dataset(X_sar_val, X_dem_val, y_val, shuffle=False)
test_dataset = create_dataset(X_sar_test, X_dem_test, y_test, shuffle=False)


def create_model(input_shape):
    sar_input = tf.keras.layers.Input(shape=input_shape)
    dem_input = tf.keras.layers.Input(shape=input_shape)

    # SAR branch
    x1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(sar_input)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D()(x1)

    x1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D()(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    # DEM branch
    x2 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(dem_input)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D()(x2)

    x2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D()(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    # Combine branches
    combined = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Dense(128, activation="relu")(combined)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[sar_input, dem_input], outputs=output)
    return model


model = create_model((128, 128, 1))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6
    ),
]


history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=50, callbacks=callbacks
)

test_results = model.evaluate(test_dataset)
print("Test Loss, Test Accuracy:", test_results)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(132)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.subplot(133)
plt.plot(history.history["lr"], label="Learning Rate")
plt.title("Learning Rate")
plt.legend()

plt.tight_layout()
plt.show()


predictions = model.predict(test_dataset)


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







"""
