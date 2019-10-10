import tensorflow as tf


def load_generator(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding="same", activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(3, 3, padding="same", activation='relu', input_shape=input_shape)
    ])
