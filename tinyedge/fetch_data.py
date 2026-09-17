"""Build a local CIFAR-10 cache (data/cifar10.npz) from the Hugging Face mirror.

`tf.keras.datasets.cifar10.load_data()` pulls from cs.toronto.edu, which is often
very slow. This script fetches the same dataset from a faster mirror and stores it
as a single .npz that `tinyedge.data.load_cifar10` picks up automatically.
"""

import argparse
import io
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
SPLITS = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
CACHE_PATH = os.path.join("data", "cifar10.npz")


def download(split, data_dir):
    path = os.path.join(data_dir, SPLITS[split])
    if not os.path.exists(path):
        url = f"{BASE_URL}/{SPLITS[split]}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def decode(parquet_path):
    table = pq.read_table(parquet_path)
    image_column = "img" if "img" in table.column_names else "image"
    images = table.column(image_column).to_pylist()
    labels = np.array(table.column("label").to_pylist(), dtype="uint8")
    x = np.stack(
        [np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB")) for img in images]
    ).astype("uint8")
    return x, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=CACHE_PATH)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    x_train, y_train = decode(download("train", args.data_dir))
    x_test, y_test = decode(download("test", args.data_dir))
    print(f"train {x_train.shape} {y_train.shape} | test {x_test.shape} {y_test.shape}")

    np.savez_compressed(
        args.output, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
