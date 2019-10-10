from pathlib import Path

import cv2
import tensorflow as tf
print(tf.__version__)

BATCH_SIZE = 4
PATCH_SIZE = (256, 256)


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


dataset_path = Path("datasets") / "gopro"
train_dataset_path = dataset_path / "train"
test_dataset_path = dataset_path / "test"

train_images_path = [str(path) for path in train_dataset_path.glob("*/sharp/*.png")]
test_images_path = [str(path) for path in test_dataset_path.glob("*/sharp/*.png")]

train_dataset = tf.data.Dataset.from_tensor_slices(train_images_path)
train_dataset = (
    train_dataset.map(  # Load sharp and blur path
        lambda path: (path, tf.strings.regex_replace(path, "sharp", "blur"))
    )
    .map(  # Read both sharp and blur files
        lambda sharp_path, blur_path: (
            tf.io.read_file(sharp_path),
            tf.io.read_file(blur_path),
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
            sharp_image, blur_image, PATCH_SIZE[0], PATCH_SIZE[1]
        )
    )
)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.shuffle(buffer_size=100)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=BATCH_SIZE // 2)

for sharps, blurs in train_dataset:
    sample_sharp, sample_blur = sharps[0], blurs[0]

    cv2.imwrite(
        "sample_blur.png",
        cv2.cvtColor((255 * sample_blur.numpy()).astype("uint8"), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        "sample_sharp.png",
        cv2.cvtColor((255 * sample_sharp.numpy()).astype("uint8"), cv2.COLOR_RGB2BGR),
    )
    break
