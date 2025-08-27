import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from collections import Counter

def load_emnist_digits():
    emnist_train = tfds.load('emnist/digits', split='train', as_supervised=True)
    emnist_test = tfds.load('emnist/digits', split='test', as_supervised=True)

    def fix_rotation(image, label):
        image = tf.image.rot90(image, k=3)
        image = tf.image.flip_left_right(image)
        return image, label

    emnist_train = emnist_train.map(fix_rotation)
    emnist_test = emnist_test.map(fix_rotation)

    y_train = [label for _, label in tfds.as_numpy(emnist_train)]
    y_test = [label for _, label in tfds.as_numpy(emnist_test)]
    
    return np.array(y_train), np.array(y_test)

if __name__ == "__main__":
    y_train, y_test = load_emnist_digits()
    all_labels = np.concatenate([y_train, y_test])
    counter = Counter(all_labels)

    print("📊 EMNIST Digit Distribution (Train + Test):")
    for digit in range(10):
        print(f"Digit {digit}: {counter[digit]}")
