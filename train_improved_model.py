import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from merge_datasets import merge_datasets

# ✅ Step 1: Load merged dataset (EMNIST + synthetic + blank)
(x_train, y_train), (x_test, y_test) = merge_datasets()

print(f"✅ Dataset Loaded:")
print(f"🟩 Training samples: {x_train.shape}, Labels: {y_train.shape}")
print(f"🟦 Test samples: {x_test.shape}, Labels: {y_test.shape}")

# ✅ Step 2: Define CNN architecture
model = Sequential([
    Input(shape=(28, 28, 1)),  # Preferred over passing input_shape to Conv2D
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Output layer for 10 digits (0–9)
])

# ✅ Step 3: Compile the model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Step 4: Callbacks
callbacks = [
    EarlyStopping(patience=2, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint("best_digit_cnn.keras", save_best_only=True, monitor='val_loss')
]

# ✅ Step 5: Train the model
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
    callbacks=callbacks
)

# ✅ Step 6: Save final model
model.save("improved_digit_cnn.h5")
print("✅ Model trained and saved as improved_digit_cnn.h5")