import tensorflow as tf


def perceptual_loss(y_true, y_pred, sample_weight, loss_model):
    return tf.reduce_mean(tf.square((loss_model(y_true), loss_model(y_pred))))
