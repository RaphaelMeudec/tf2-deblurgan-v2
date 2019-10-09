from pathlib import Path

import tensorflow as tf


print(tf.__version__)

BATCH_SIZE = 4

dataset_path = Path("datasets") / "gopro"
train_dataset_path = dataset_path / "train"
test_dataset_path = dataset_path / "test"

train_images_path = [str(path) for path in train_dataset_path.glob("*/sharp/*.png")]
test_images_path = [str(path) for path in test_dataset_path.glob("*/sharp/*.png")]

train_dataset = tf.data.Dataset.from_tensor_slices(train_images_path)
train_dataset = train_dataset.map(lambda path: (path, tf.strings.regex_replace(path, 'sharp', 'blur')))
train_dataset = train_dataset.map(lambda sharp_path, blur_path: (tf.io.read_file(sharp_path), tf.io.read_file(blur_path)))
train_dataset = train_dataset.map(lambda sharp_file, blur_file: (tf.image.decode_png(sharp_file, channels=3), tf.image.decode_png(blur_file, channels=3)))
train_dataset = train_dataset.map(lambda sharp_image, blur_image: (
    tf.image.convert_image_dtype(sharp_image, tf.float32),
    tf.image.convert_image_dtype(blur_image, tf.float32)
))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.shuffle(buffer_size=100)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=BATCH_SIZE//2)



for images in train_dataset:
    print(len(images))
    print(images[0].shape)
    break
