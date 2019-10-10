import tensorflow as tf

from dataset import load_dataset
from model import load_generator

print(tf.__version__)

BATCH_SIZE = 4
PATCH_SIZE = (256, 256)
INPUT_SHAPE = (*PATCH_SIZE, 3)

generator = load_generator(INPUT_SHAPE)
generator(tf.zeros((1, *INPUT_SHAPE)))
generator.summary()

optimizer = tf.keras.optimizers.Adam(1e-4)

dataset = load_dataset("gopro", patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, mode="train")


@tf.function
def training_loop(model, sharp_images, blur_images):
    with tf.GradientTape() as tape:
        blurred = tf.cast(blur_images, tf.float32)
        sharp = tf.cast(sharp_images, tf.float32)

        deblurred = model(blurred, training=True)

        loss = tf.reduce_sum(tf.abs(deblurred - sharp))

    grads = tape.gradient(loss, model.trainable_weights)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss


for index, (sharp_images, blur_images) in enumerate(dataset):
    print(training_loop(generator, sharp_images, blur_images))

    if index > 3:
        break
