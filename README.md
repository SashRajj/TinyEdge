# TinyEdge

Quantized CNN inference for edge AI: train a small image classifier, convert it to
TensorFlow Lite, apply INT8 post-training quantization, and measure exactly what that
costs in accuracy and buys in model size and latency.

Every number below is produced by `python -m tinyedge.benchmark` on the full
10,000-image CIFAR-10 test set — nothing is estimated.

## Results

CIFAR-10 test set, 290k-parameter CNN, single-thread TFLite CPU interpreter on an
Apple Silicon laptop (M-series, macOS 14), TensorFlow 2.21.

| Model | Top-1 accuracy | Size | Latency (mean) | Latency (p95) | vs FP32 TFLite |
|---|---|---|---|---|---|
| Keras FP32 (`.keras`) | 88.13% | 3497.1 KiB | — | — | — |
| TFLite FP32 | 88.13% | 1132.4 KiB | 1.17 ms | 1.25 ms | baseline |
| TFLite dynamic-range | 88.10% | 296.8 KiB | 0.28 ms | 0.29 ms | 73.8% smaller, 4.2x faster, −0.03 pp |
| **TFLite INT8 (full)** | **88.13%** | **300.6 KiB** | **0.25 ms** | **0.26 ms** | **73.5% smaller, 4.7x faster, −0.00 pp** |

Headline: **73.5% smaller and 4.7x faster at zero net accuracy cost.**

The INT8 model scoring exactly the same as FP32 is a coincidence worth being precise
about, because "identical accuracy" usually means something is wrong. It isn't: the two
models disagree on 75 of the 10,000 test images. Quantization flips 31 wrong answers to
right and 31 right answers to wrong, so the totals tie at 8813 correct. Top-1 agreement
between them is 99.25%.

Note that `.keras` and `.tflite` sizes are not directly comparable — the Keras file
carries optimizer state and training metadata. The meaningful comparison is TFLite FP32
against TFLite INT8, both of which are deployment artifacts.

## What the pipeline does

1. **Train** a VGG-style CNN on CIFAR-10 (`tinyedge/train.py`): three conv blocks,
   batch norm, global average pooling, cosine-decayed Adam, random flip and 4-pixel
   shift augmentation. 40 epochs, 289,642 parameters, 88.13% test accuracy.
2. **Convert** to three TFLite flatbuffers (`tinyedge/convert.py`):
   - `fp32` — plain conversion; the baseline for every comparison.
   - `dynamic` — dynamic-range quantization: INT8 weights, float activations, no
     calibration data required.
   - `int8` — full integer quantization: INT8 weights *and* activations, with INT8
     input and output tensors, calibrated on 300 unlabelled training images.
3. **Benchmark** all three (`tinyedge/benchmark.py`): top-1 accuracy on the full test
   set, file size on disk, and single-image single-thread latency (20 warmup runs,
   200 timed runs, mean/p50/p95). Results land in `results/benchmark.json`.

## Design notes

**The model only uses ops with INT8 kernels.** Full integer quantization falls back to
float — silently, and with it most of the benefit — if the graph contains an op the
INT8 path can't handle. Sticking to conv2d, batch norm (folded away at conversion),
ReLU, max pooling, global average pooling and one dense layer keeps the graph fully
quantizable end to end.

**Global average pooling instead of flatten + dense.** A flatten into a wide dense layer
would put most of the parameters in one place and dominate the model size; GAP keeps the
network at 290k parameters, so the 73.5% reduction reflects the whole network rather
than one fat layer.

**Calibration defines the accuracy cost.** The representative dataset sets the scale and
zero point for every activation tensor. Too few samples and rare activations clip, which
is where most INT8 accuracy loss actually comes from. 300 images were enough here; the
count is exposed as `--calibration-samples` for experimenting.

**Why INT8 is barely smaller than dynamic-range.** Both store INT8 weights, and weights
are nearly all of the file. Full INT8 adds per-tensor quantization parameters for
activations, which is why it is fractionally *larger* on disk. Its advantage is that
activations are integers too, so it runs on integer-only hardware (microcontrollers,
NPUs, DSPs) where the dynamic-range model cannot.

## Layout

```
tinyedge/
  data.py        CIFAR-10 loading, normalization, calibration sampler
  model.py       the CNN
  train.py       FP32 training
  convert.py     TFLite conversion: fp32 / dynamic / int8
  benchmark.py   accuracy, size and latency comparison
  fetch_data.py  optional dataset download from a faster mirror
models/          .tflite deployment artifacts (committed)
results/         benchmark.json
```

## Quickstart

```bash
make setup                      # venv + dependencies
python -m tinyedge.fetch_data   # optional, see below
make train                      # ~90 min on a laptop CPU (40 epochs)
make convert
make benchmark
```

Every step is a module with `--help`: `python -m tinyedge.train --epochs 10` for a quick
run, `python -m tinyedge.benchmark --test-samples 1000` for a fast benchmark.

`tinyedge.fetch_data` is optional. `tf.keras.datasets.cifar10` downloads from
cs.toronto.edu, which can be extremely slow; the script pulls the same dataset from a
Hugging Face mirror into `data/cifar10.npz`, which `load_cifar10()` then prefers.

## Limitations

- Latency is measured with the TFLite CPU interpreter pinned to one thread on a laptop.
  It is a like-for-like comparison between three models on identical hardware, not a
  prediction of throughput on any particular microcontroller or NPU.
- The 4.7x speedup partly reflects XNNPACK's INT8 kernels on this CPU. The size
  reduction is the portable result: roughly 4x on any target, since it follows directly
  from 32-bit weights becoming 8-bit.
- Post-training quantization only. Quantization-aware training would matter more for a
  model where PTQ actually hurts accuracy; here there was nothing to recover.
- Single training run, fixed seed. No error bars across seeds.

## License

MIT
