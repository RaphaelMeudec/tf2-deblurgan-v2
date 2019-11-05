import numpy as np
import tensorflow as tf

from dataset import select_patch
from model import FPNInception

blur_image_path = "/home/raph/Projects/tf2-deblurgan-v2/datasets/gopro/test/GOPR0384_11_00/blur/000020.png"
sharp_image_path = "/home/raph/Projects/tf2-deblurgan-v2/datasets/gopro/test/GOPR0384_11_00/sharp/000020.png"

model_path = "/home/raph/Projects/tf2-deblurgan-v2/best_model.h5"


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - 0.5) * 2

    return image


blur_image = load_image(blur_image_path)
sharp_image = load_image(sharp_image_path)
sharp_image, blur_image = select_patch(sharp_image, blur_image, 256, 256)

# model = tf.keras.models.load_model(model_path)
model = FPNInception(num_filters=128, num_filters_fpn=256)
model(np.array([blur_image]))
model.load_weights(model_path)


def deprocess_image(image):
    image = image / 2 + 0.5
    return image


deblurred = model(np.array([blur_image]))[0]

import cv2

cv2.imwrite(
    "sharp.png",
    cv2.cvtColor(
        (deprocess_image(sharp_image) * 255).numpy().astype("uint8"), cv2.COLOR_BGR2RGB
    ),
)
cv2.imwrite(
    "deblurred.png",
    cv2.cvtColor(
        (deprocess_image(deblurred) * 255).numpy().astype("uint8"), cv2.COLOR_BGR2RGB
    ),
)
cv2.imwrite(
    "blurred.png",
    cv2.cvtColor(
        (deprocess_image(blur_image) * 255).numpy().astype("uint8"), cv2.COLOR_BGR2RGB
    ),
)
