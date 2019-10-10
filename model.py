import tensorflow as tf


def load_generator(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding="same", activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(3, 3, padding="same", activation='relu')
    ])

def load_discriminator(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
