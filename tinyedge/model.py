"""A small VGG-style CNN sized for edge deployment."""

import tensorflow as tf
from tensorflow.keras import layers

from tinyedge.data import INPUT_SHAPE, NUM_CLASSES


def build_model(width=32, dropout=0.3):
    """Three conv blocks, batch norm, global average pooling head.

    Only ops with well-supported INT8 TFLite kernels are used (conv2d, relu,
    max pool, global average pool, dense), so the graph quantizes end to end.
    """
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = inputs
    for i, filters in enumerate((width, width * 2, width * 4)):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout * (i + 1) / 3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="tinyedge_cnn")
