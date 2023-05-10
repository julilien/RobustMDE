import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


class TrimLoss(LossFunctionWrapper):
    """
    Trimmed MAE loss as described by Ranftl et. al, 2020.
    """

    def __init__(self, batch_size, c_param=0.8, loss_fn=None, name="trim"):
        if loss_fn is None:
            loss_fn = self.loss
        super(TrimLoss, self).__init__(fn=loss_fn, name=name)

        self.c_param = c_param
        self.batch_size = batch_size

    def loss(self, y_true, y_pred):
        residuals = tf.math.abs(y_true - y_pred)

        residuals = tf.reshape(residuals, [self.batch_size, -1])

        sorted_residuals = tf.sort(residuals, axis=-1, direction='ASCENDING')[:,
                           :tf.cast(self.c_param * tf.cast(tf.shape(residuals)[1], tf.float32), tf.int32)]

        return tf.reduce_mean(sorted_residuals, axis=-1)
