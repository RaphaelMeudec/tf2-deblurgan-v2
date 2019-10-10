from pathlib import Path

import cv2
import tensorflow as tf


def select_patch(sharp, blur, patch_size_x, patch_size_y):
    offset_height = 1
    offset_width = 1
    return (
        tf.image.crop_to_bounding_box(
            sharp, offset_height, offset_width, patch_size_x, patch_size_y
        ),
        tf.image.crop_to_bounding_box(
            blur, offset_height, offset_width, patch_size_x, patch_size_y
        ),
    )


def get_dataset_path(dataset_name):
    return Path("datasets") / dataset_name


def load_dataset(dataset_name, patch_size, batch_size, mode="train"):
    dataset_path = get_dataset_path(dataset_name)
    subset_dataset_path = dataset_path / mode

    images_path = [str(path) for path in subset_dataset_path.glob("*/sharp/*.png")]

    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = (
        dataset.map(
            lambda path: (path, tf.strings.regex_replace(path, "sharp", "blur"))
        )
        .map(  # Read both sharp and blur files
            lambda sharp_path, blur_path: (
                tf.io.read_file(sharp_path),
                tf.io.read_file(blur_path)
            )
        )
            .map(  # Decode as png both sharp and blur files
            lambda sharp_file, blur_file: (
                tf.image.decode_png(sharp_file, channels=3),
                tf.image.decode_png(blur_file, channels=3),
            )
        )
            .map(  # Convert to float32 both sharp and blur files
            lambda sharp_image, blur_image: (
                tf.image.convert_image_dtype(sharp_image, tf.float32),
                tf.image.convert_image_dtype(blur_image, tf.float32),
            )
        )
            .map(  # Select subset of the image
            lambda sharp_image, blur_image: select_patch(
                sharp_image, blur_image, patch_size[0], patch_size[1]
            )
        )
    )

    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size // 2)

    return dataset


if __name__ == "__main__":
    train_dataset = load_dataset("gopro", patch_size=(128, 128), batch_size=16, mode="train")
    for sharps, blurs in train_dataset.take(1):
        sample_sharp, sample_blur = sharps[0], blurs[0]

        cv2.imwrite(
            "sample_blur.png",
            cv2.cvtColor((255 * sample_blur.numpy()).astype("uint8"), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            "sample_sharp.png",
            cv2.cvtColor((255 * sample_sharp.numpy()).astype("uint8"), cv2.COLOR_RGB2BGR),
        )
