import keras
import tensorflow as tf
import numpy as np


n_classes = 10
window_length = 15

def multinom(X, y, )
    model = keras.Sequential([
        keras.layers.Dense(n_classes, activation=tf.nn.softmax,
                           input_shape=(n_classes * window_length))
    ])


