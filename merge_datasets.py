import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical

# Step 1: Load EMNIST Digits
def load_emnist_digits():
    emnist_train = tfds.load('emnist/digits', split='train', as_supervised=True)
    emnist_test = tfds.load('emnist/digits', split='test', as_supervised=True)

    def fix_rotation(image, label):
        image = tf.image.rot90(image, k=3)
        image = tf.image.flip_left_right(image)
        return image, label

    emnist_train = emnist_train.map(lambda x, y: fix_rotation(x, y))
    emnist_test = emnist_test.map(lambda x, y: fix_rotation(x, y))

    x_train, y_train = [], []
    for image, label in tfds.as_numpy(emnist_train):
        x_train.append(image)
        y_train.append(label)

    x_test, y_test = [], []
    for image, label in tfds.as_numpy(emnist_test):
        x_test.append(image)
        y_test.append(label)

    x_train = np.array(x_train).reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = np.array(x_test).reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

# Step 2: Load synthetic digits generated earlier
def load_synthetic_digits(path="synthetic_digits"):
    x, y = [], []

    for digit in range(10):
        digit_dir = os.path.join(path, str(digit))
        if os.path.exists(digit_dir):
            for fname in os.listdir(digit_dir):
                img_path = os.path.join(digit_dir, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (28, 28)) / 255.0
                    x.append(img)
                    y.append(digit)

    # ✅ NEW: Load 0_blank as more label 0s
    blank_dir = os.path.join(path, "0_blank")
    if os.path.exists(blank_dir):
        for fname in os.listdir(blank_dir):
            img_path = os.path.join(blank_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (28, 28)) / 255.0
                x.append(img)
                y.append(0)  # still label as 0 (blank cell)

    x = np.array(x).reshape(-1, 28, 28, 1)
    y = np.array(y)
    return x, y

# Step 3: Merge EMNIST and synthetic data
def merge_datasets():
    (x_emnist_train, y_emnist_train), (x_emnist_test, y_emnist_test) = load_emnist_digits()
    x_syn, y_syn = load_synthetic_digits()

    x_train = np.concatenate([x_emnist_train, x_syn], axis=0)
    y_train = np.concatenate([y_emnist_train, y_syn], axis=0)

    # One-hot encode for CNN classification
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_emnist_test, 10)

    return (x_train, y_train), (x_emnist_test, y_test)

# Step 4: Run the merge and save to disk
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = merge_datasets()
    print("✅ Final Merged Training Set:", x_train.shape, y_train.shape)
    print("✅ Final Merged Testing Set:", x_test.shape, y_test.shape)

    # 🔥 Save datasets to disk
    os.makedirs("merged_dataset", exist_ok=True)
    np.save("merged_dataset/x_train.npy", x_train)
    np.save("merged_dataset/y_train.npy", y_train)
    np.save("merged_dataset/x_test.npy", x_test)
    np.save("merged_dataset/y_test.npy", y_test)
    print("✅ Saved merged dataset to 'merged_dataset/' folder.")
