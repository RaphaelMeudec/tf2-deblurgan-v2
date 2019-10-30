from pathlib import Path

import cv2
import tensorflow as tf


def select_patch(sharp, blur, patch_size_x, patch_size_y):
    stack = tf.stack([sharp, blur], axis=0)
    patches = tf.image.random_crop(stack, size=[2, patch_size_x, patch_size_y, 3])
    return (patches[0], patches[1])


def get_dataset_path(dataset_name):
    return Path("datasets") / dataset_name


def load_dataset(
    dataset_name, patch_size, batch_size, mode="train", shuffle=False, cache=False
):
    dataset_path = get_dataset_path(dataset_name)
    subset_dataset_path = dataset_path / mode

    images_path = [str(path) for path in subset_dataset_path.glob("*/sharp/*.png")]

    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = (
        dataset.map(
            lambda path: (path, tf.strings.regex_replace(path, "sharp", "blur")),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(  # Read both sharp and blur files
            lambda sharp_path, blur_path: (
                tf.io.read_file(sharp_path),
                tf.io.read_file(blur_path),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(  # Decode as png both sharp and blur files
            lambda sharp_file, blur_file: (
                tf.image.decode_png(sharp_file, channels=3),
                tf.image.decode_png(blur_file, channels=3),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )
    if cache:
        dataset = dataset.cache()
    dataset = (
        dataset.map(  # Convert to float32 both sharp and blur files
            lambda sharp_image, blur_image: (
                tf.image.convert_image_dtype(sharp_image, tf.float32),
                tf.image.convert_image_dtype(blur_image, tf.float32),
            )
        )
        .map(  # Load images between [-1, 1] instead of [0, 1]
            lambda sharp_image, blur_image: (
                (sharp_image - 0.5) * 2,
                (blur_image - 0.5) * 2,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(  # Select subset of the image
            lambda sharp_image, blur_image: select_patch(
                sharp_image, blur_image, patch_size[0], patch_size[1]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    train_dataset = load_dataset(
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
