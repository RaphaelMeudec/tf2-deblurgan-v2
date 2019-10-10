import tensorflow as tf

from dataset import load_dataset
from model import load_discriminator, load_generator

print(tf.__version__)

BATCH_SIZE = 4
PATCH_SIZE = (256, 256)
INPUT_SHAPE = (*PATCH_SIZE, 3)


class Trainer:
    def __init__(self, dataset, input_shape):
        self.dataset = dataset
        self.input_shape = input_shape

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.generator = load_generator(input_shape)
        self.generator(tf.zeros((1, *input_shape)))
        self.generator.summary()
        self.discriminator = load_discriminator(input_shape)
        self.discriminator(tf.zeros((1, *input_shape)))
        self.discriminator.summary()

    def train(self, num_batch):
        for index, (sharp_images, blur_images) in enumerate(dataset):
            if index > num_batch:
                break

            loss = self.training_loop(sharp_images, blur_images)
            print(loss)

    @tf.function
    def training_loop(self, sharp_images, blur_images):
        with tf.GradientTape() as tape:
            blurred = tf.cast(blur_images, tf.float32)
            sharp = tf.cast(sharp_images, tf.float32)

            deblurred = self.generator(blurred, training=True)

            loss = tf.reduce_sum(tf.abs(deblurred - sharp))

        grads = tape.gradient(loss, self.generator.trainable_weights)

        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return loss


if __name__ == "__main__":
    dataset = load_dataset("gopro", patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, mode="train")

    trainer = Trainer(dataset, INPUT_SHAPE)
    trainer.train(3)
