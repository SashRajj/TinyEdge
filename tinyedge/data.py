"""CIFAR-10 loading and preprocessing."""

import os

import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


CACHE_PATH = os.path.join("data", "cifar10.npz")


def load_cifar10():
    """Return (x_train, y_train), (x_test, y_test) with images as float32 in [0, 1].

    Uses the local cache written by `python -m tinyedge.fetch_data` when present,
    otherwise falls back to the standard Keras download.
    """
    if os.path.exists(CACHE_PATH):
        cache = np.load(CACHE_PATH)
        x_train, y_train = cache["x_train"], cache["y_train"]
        x_test, y_test = cache["x_test"], cache["y_test"]
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train.flatten()), (x_test, y_test.flatten())


def representative_dataset(x_train, num_samples=300, seed=0):
    """Yield single-image batches used to calibrate activation ranges for INT8 PTQ."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_train), size=num_samples, replace=False)

    def generator():
        for i in idx:
            yield [x_train[i : i + 1]]

    return generator
