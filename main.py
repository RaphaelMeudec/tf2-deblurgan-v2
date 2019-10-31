from functools import partial
from pathlib import Path

import tensorflow as tf

from dataset import load_dataset
from losses import perceptual_loss
from model import FPNInception, NLayerDiscriminator

print(tf.__version__)

BATCH_SIZE = 4
PATCH_SIZE = (256, 256)
INPUT_SHAPE = (*PATCH_SIZE, 3)


class CNNTrainer:
    def __init__(
        self, dataset, validation_dataset, input_shape, fit_method_arguments=None
    ):
        self.dataset = dataset
        self.validation_dataset = validation_dataset

        fit_method_arguments = fit_method_arguments if fit_method_arguments else {}
        self.fit_method_arguments = {
            "validation_steps": 1000,
            "steps_per_epoch": 10000,
            "epochs": 100,
            **fit_method_arguments,
        }

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

    def train(self):
        callbacks = [
            tf.keras.callbacks.TensorBoard(profile_batch=3),
            tf.keras.callbacks.ModelCheckpoint(
                filepath="best_model.h5", monitor="val_loss", save_best_only=True
            ),
        ]

        self.model.fit(
            self.dataset,
            validation_data=self.validation_dataset,
            callbacks=callbacks,
            **self.fit_method_arguments,
        )

        # for index, (x, y) in enumerate(self.dataset):
        #     loss = self.training_step(x, y)
        #
        #     if index > 100:
        #         break

    @tf.function
    def training_step(self, x, y):
        with tf.GradientTape() as tape:
            deblurred = self.model(x)
            loss = perceptual_loss(deblurred, y, sample_weight=None, loss_model=self.loss_model)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return loss

if __name__ == "__main__":
    dataset = load_dataset(
        "gopro",
        patch_size=PATCH_SIZE,
        batch_size=BATCH_SIZE,
        mode="train",
        shuffle=True,
        cache=True,
    )
    dataset_length = len(
        [el for el in (Path("datasets") / "gopro" / "train").rglob("*/sharp/*.png")]
    )

    validation_dataset = load_dataset(
        "gopro", patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, mode="test"
    )

    tf.summary.trace_on(profiler=True)

    trainer = CNNTrainer(
        dataset,
        validation_dataset,
        INPUT_SHAPE,
        {
            "steps_per_epoch": 100,
            "validation_steps": 10,
            "epochs": 1
        },
    )
    trainer.train()

    tf.summary.trace_export(name="cnntrainer", profiler_outdir="./profiling")
