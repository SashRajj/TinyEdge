"""Convert the trained Keras model to TFLite: FP32 baseline, dynamic-range, full INT8."""

import argparse
import os
import shutil
import tempfile

import tensorflow as tf

from tinyedge.data import load_cifar10, representative_dataset


def _saved_model_dir(keras_path, export_dir):
    """Keras 3 models convert most reliably through an exported SavedModel."""
    model = tf.keras.models.load_model(keras_path)
    model.export(export_dir)
    return export_dir


def convert_fp32(saved_model_dir):
    return tf.lite.TFLiteConverter.from_saved_model(saved_model_dir).convert()


def convert_dynamic(saved_model_dir):
    """INT8 weights, float activations — no calibration data required."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_int8(saved_model_dir, rep_data):
    """Full integer quantization: INT8 weights, activations and INT8 input/output."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/tinyedge_fp32.keras")
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--calibration-samples", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    (x_train, _), _ = load_cifar10()
    rep_data = representative_dataset(x_train, num_samples=args.calibration_samples)

    export_dir = tempfile.mkdtemp(prefix="tinyedge_saved_model_")
    try:
        _saved_model_dir(args.model, export_dir)
        variants = {
            "fp32": convert_fp32(export_dir),
            "dynamic": convert_dynamic(export_dir),
            "int8": convert_int8(export_dir, rep_data),
        }
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)

    for name, flatbuffer in variants.items():
        path = os.path.join(args.outdir, f"tinyedge_{name}.tflite")
        with open(path, "wb") as f:
            f.write(flatbuffer)
        print(f"{name:>8}: {len(flatbuffer) / 1024:8.1f} KiB -> {path}")


if __name__ == "__main__":
    main()
