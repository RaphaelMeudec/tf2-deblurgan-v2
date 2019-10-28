from functools import partial

import tensorflow as tf

from dataset import load_dataset
from losses import perceptual_loss
from model import FPNInception, NLayerDiscriminator

print(tf.__version__)

BATCH_SIZE = 4
PATCH_SIZE = (256, 256)
INPUT_SHAPE = (*PATCH_SIZE, 3)


class CNNTrainer:
    def __init__(self, dataset, validation_dataset, input_shape):
        self.dataset = dataset
        self.validation_dataset = validation_dataset

        self.input_shape = input_shape

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        vgg = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        self.loss_model = tf.keras.models.Model(
            inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
        )

        self.loss = partial(perceptual_loss, loss_model=self.loss_model)

        self.model = FPNInception(num_filters=128, num_filters_fpn=256)
        self.model(tf.random.uniform((4, *input_shape)))
        self.model.summary()

        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def train(self, num_epochs):
        callbacks = [
            tf.keras.callbacks.TensorBoard(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath="best_model.h5", monitor="val_loss", save_best_only=True
            ),
        ]

        self.model.fit(
            self.dataset,
            validation_data=self.validation_dataset,
            validation_steps=10,
            epochs=num_epochs,
            steps_per_epoch=10,
            callbacks=callbacks,
        )


class GANTrainer:
    def __init__(self, dataset, input_shape):
        self.dataset = dataset
        self.input_shape = input_shape

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.generator = FPNInception(num_filters=128, num_filters_fpn=256)
        self.generator(tf.zeros((1, *input_shape)))
        self.generator.summary()

        self.discriminator = NLayerDiscriminator(ndf=64, n_layers=3)
        self.discriminator(tf.zeros((1, *input_shape)))
        self.discriminator.summary()

    def train(self, num_batch):
        for index, (sharp_images, blur_images) in enumerate(dataset):
            if index > num_batch:
                break

            generator_loss, discriminator_loss = self.training_loop(
                sharp_images, blur_images
            )
            # TODO: Add callbacks
            print(generator_loss.numpy(), discriminator_loss.numpy())

    @tf.function
    def training_loop(self, sharp_images, blur_images):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            blurred = tf.cast(blur_images, tf.float32)
            sharp = tf.cast(sharp_images, tf.float32)

            deblurred = self.generator(blurred, training=True)

            discriminated_deblurred = self.discriminator(deblurred, training=True)
            discriminated_sharp = self.discriminator(sharp_images, training=True)

            # TODO: Specify correct losses
            generator_loss = tf.reduce_sum(tf.abs(deblurred - sharp))
            discriminator_loss = tf.reduce_sum(
                tf.abs(discriminated_deblurred)
            ) - tf.reduce_sum(tf.abs(discriminated_sharp))

        generator_grads = generator_tape.gradient(
            generator_loss, self.generator.trainable_weights
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_grads, self.generator.trainable_weights)
        )

        discriminator_grads = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_weights
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_grads, self.discriminator.trainable_weights)
        )

        return generator_loss, discriminator_loss


if __name__ == "__main__":
    dataset = load_dataset(
        "gopro", patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, mode="train"
    )

    validation_dataset = load_dataset(
        "gopro", patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, mode="test"
    )

    trainer = CNNTrainer(dataset, validation_dataset, INPUT_SHAPE)
    trainer.train(3)
