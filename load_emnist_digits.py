import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

# Load EMNIST digits split
ds_train, ds_test = tfds.load(
    'emnist/digits',
    split=['train', 'test'],
    as_supervised=True
)

def ds_to_numpy(dataset):
    images, labels = [], []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

x_train_emnist, y_train_emnist = ds_to_numpy(ds_train)
x_test_emnist, y_test_emnist = ds_to_numpy(ds_test)

# Normalize and reshape
x_train_emnist = x_train_emnist.astype('float32') / 255.0
x_test_emnist = x_test_emnist.astype('float32') / 255.0
x_train_emnist = x_train_emnist.reshape(-1, 28, 28, 1)
x_test_emnist = x_test_emnist.reshape(-1, 28, 28, 1)

print("✅ EMNIST Digits Loaded:")
print("Train:", x_train_emnist.shape, y_train_emnist.shape)
print("Test :", x_test_emnist.shape, y_test_emnist.shape)
