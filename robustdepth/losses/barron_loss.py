import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
import tensorflow.keras.backend as keras_backend


class BarronLoss(LossFunctionWrapper):
    def __init__(self, alpha_param, c_param, loss_fn=None, name="barron"):
        if loss_fn is None:
            loss_fn = self.loss
        super(BarronLoss, self).__init__(fn=loss_fn, name=name)

        # So far, this variant only works for positive alphas
        self.alpha_param = alpha_param
        # Must be positive
        self.c_param = c_param

    def loss(self, y_true, y_pred):
        alpha_2 = tf.cast(keras_backend.clip(tf.math.abs(self.alpha_param - 2.), 1e-7, None), dtype=y_true.dtype)
        clipped_alpha = tf.cast(keras_backend.clip(self.alpha_param, 1e-7, None), dtype=y_true.dtype)

        factor = alpha_2 / clipped_alpha
        x = tf.cast(y_true - y_pred, dtype=y_true.dtype)

        return tf.cast(factor, y_true.dtype) * (tf.math.pow(
            tf.math.pow(x / tf.cast(self.c_param, y_true.dtype), tf.constant(2., dtype=y_true.dtype)) / alpha_2 + 1.,
            clipped_alpha / tf.constant(2., dtype=y_true.dtype)) - tf.constant(1., dtype=y_true.dtype))
