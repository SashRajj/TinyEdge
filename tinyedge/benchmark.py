"""Benchmark the FP32 and quantized models: accuracy, size on disk, latency."""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from tinyedge.data import load_cifar10

try:  # LiteRT is the current home of the TFLite runtime; tf.lite still works today.
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    Interpreter = tf.lite.Interpreter


def quantize_input(x, details):
    """Map float [0, 1] images into the model's input dtype (int8 for full PTQ)."""
    dtype = details["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zero_point = details["quantization"]
        x = np.round(x / scale + zero_point)
        info = np.iinfo(dtype)
        x = np.clip(x, info.min, info.max)
    return x.astype(dtype)


def evaluate_tflite(path, x_test, y_test, num_threads=1):
    interpreter = Interpreter(model_path=path, num_threads=num_threads)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    correct = 0
    for i in range(len(x_test)):
        interpreter.set_tensor(inp["index"], quantize_input(x_test[i : i + 1], inp))
        interpreter.invoke()
        logits = interpreter.get_tensor(out["index"])
        correct += int(np.argmax(logits) == y_test[i])
    return correct / len(x_test)


def measure_latency(path, x_test, runs=200, warmup=20, num_threads=1):
    """Single-image, single-thread latency — the edge inference case."""
    interpreter = Interpreter(model_path=path, num_threads=num_threads)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    sample = quantize_input(x_test[:1], inp)

    for _ in range(warmup):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()

    timings = []
    for i in range(runs):
        sample = quantize_input(x_test[i % len(x_test) : i % len(x_test) + 1], inp)
        interpreter.set_tensor(inp["index"], sample)
        start = time.perf_counter()
        interpreter.invoke()
        timings.append((time.perf_counter() - start) * 1000.0)
    timings = np.array(timings)
    return {
        "mean_ms": float(timings.mean()),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras-model", default="models/tinyedge_fp32.keras")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--variants", nargs="+", default=["fp32", "dynamic", "int8"])
    parser.add_argument("--test-samples", type=int, default=10_000)
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()

    _, (x_test, y_test) = load_cifar10()
    x_test, y_test = x_test[: args.test_samples], y_test[: args.test_samples]

    results = {}
    keras_model = tf.keras.models.load_model(args.keras_model)
    keras_preds = keras_model.predict(x_test, verbose=0).argmax(axis=1)
    keras_acc = float((keras_preds == y_test).mean())
    results["keras_fp32"] = {
        "accuracy": float(keras_acc),
        "size_kib": os.path.getsize(args.keras_model) / 1024,
        "latency": None,
    }
    print(f"{'keras_fp32':>10}: acc {keras_acc * 100:5.2f}%")

    for variant in args.variants:
        path = os.path.join(args.models_dir, f"tinyedge_{variant}.tflite")
        acc = evaluate_tflite(path, x_test, y_test, args.threads)
        latency = measure_latency(path, x_test, args.latency_runs, num_threads=args.threads)
        results[variant] = {
            "accuracy": float(acc),
            "size_kib": os.path.getsize(path) / 1024,
            "latency": latency,
        }
        print(
            f"{variant:>10}: acc {acc * 100:5.2f}%  "
            f"size {results[variant]['size_kib']:7.1f} KiB  "
            f"latency {latency['mean_ms']:5.2f} ms (p95 {latency['p95_ms']:5.2f} ms)"
        )

    baseline = results["fp32"]
    for variant in args.variants:
        r = results[variant]
        r["size_reduction_vs_fp32_pct"] = 100 * (1 - r["size_kib"] / baseline["size_kib"])
        r["accuracy_drop_vs_fp32_pp"] = 100 * (baseline["accuracy"] - r["accuracy"])
        r["speedup_vs_fp32"] = baseline["latency"]["mean_ms"] / r["latency"]["mean_ms"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")

    int8 = results["int8"]
    print(
        f"INT8 vs FP32 TFLite: {int8['size_reduction_vs_fp32_pct']:.1f}% smaller, "
        f"{int8['accuracy_drop_vs_fp32_pp']:+.2f} pp accuracy, "
        f"{int8['speedup_vs_fp32']:.2f}x speed"
    )


if __name__ == "__main__":
    main()
