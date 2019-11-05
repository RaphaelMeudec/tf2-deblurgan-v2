from pathlib import Path

import cv2
import tensorflow as tf


def select_patch(sharp, blur, patch_size_x, patch_size_y):
    stack = tf.stack([sharp, blur], axis=0)
    patches = tf.image.random_crop(stack, size=[2, patch_size_x, patch_size_y, 3])
    return (patches[0], patches[1])


def get_dataset_path(dataset_name):
    return Path("datasets") / dataset_name


class IndependantDataLoader:
    def image_dataset(self, images_paths):
        dataset = tf.data.Dataset.from_tensor_slices(images_paths)
        dataset = (
            dataset.map(
                tf.io.read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .map(tf.image.decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(
                lambda x: tf.image.convert_image_dtype(x, tf.float32),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .map(
                lambda x: (x - 0.5) * 2,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )

        return dataset

    def load(
        self,
        dataset_name,
        mode="train",
        batch_size=4,
        patch_size=(256, 256),
        shuffle=False,
    ):
        dataset_path = get_dataset_path(dataset_name) / mode
        sharp_images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
        blur_images_path = [path.replace("sharp", "blur") for path in sharp_images_path]

        sharp_dataset = self.image_dataset(sharp_images_path).cache()
        blur_dataset = self.image_dataset(blur_images_path).cache()

        dataset = tf.data.Dataset.zip((sharp_dataset, blur_dataset))
        if shuffle:
            dataset = dataset.shuffle()

        dataset = dataset.map(
            lambda sharp_image, blur_image: select_patch(
                sharp_image, blur_image, patch_size[0], patch_size[1]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == "__main__":
    train_dataset = IndependantDataLoader().load(
        "gopro", patch_size=(128, 128), batch_size=16, mode="train"
    )
    for sharps, blurs in train_dataset.take(1):
        sample_sharp, sample_blur = sharps[0], blurs[0]

        sample_sharp = (sample_sharp / 2) + 0.5
        sample_blur = (sample_blur / 2) + 0.5

        cv2.imwrite(
            "sample_blur.png",
            cv2.cvtColor(
                (255 * sample_blur.numpy()).astype("uint8"), cv2.COLOR_RGB2BGR
            ),
        )
        cv2.imwrite(
            "sample_sharp.png",
            cv2.cvtColor(
                (255 * sample_sharp.numpy()).astype("uint8"), cv2.COLOR_RGB2BGR
            ),
        )
