"""Train the FP32 baseline CNN on CIFAR-10."""

import argparse
import os

import tensorflow as tf

from tinyedge.data import load_cifar10
from tinyedge.model import build_model

AUTOTUNE = tf.data.AUTOTUNE


def augment(image, label):
    """Standard CIFAR augmentation: random flip plus a 4-pixel shift."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=(32, 32, 3))
    return image, label


def make_datasets(x_train, y_train, x_test, y_test, batch_size):
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10_000)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train_ds, test_ds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--output", default="models/tinyedge_fp32.keras")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    train_ds, test_ds = make_datasets(x_train, y_train, x_test, y_test, args.batch_size)

    model = build_model(width=args.width)
    model.summary()

    steps = args.epochs * (len(x_train) // args.batch_size)
    schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, verbose=2)

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nFP32 Keras test accuracy: {acc * 100:.2f}%  (loss {loss:.4f})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"Saved Keras model to {args.output}")


if __name__ == "__main__":
    main()
