import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


class BerhuLoss(LossFunctionWrapper):
    """
    As described in Laina et. al, 2016: Deeper Depth Prediction with Fully Convolutional Residual Networks.
    """

    def __init__(self, c_param=0.2):
        super(BerhuLoss, self).__init__(fn=self.loss, name="berhu_loss")
        self.c_param = c_param

    def loss(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        y_true = tf.reshape(y_true, (-1, tf.reduce_prod(tf.shape(y_true)[1:])))
        y_pred = tf.reshape(y_pred, (-1, tf.reduce_prod(tf.shape(y_pred)[1:])))

        diff = y_pred - y_true
        abs_diff = tf.math.abs(diff)
        c = self.c_param * tf.math.reduce_max(abs_diff, axis=[-1])
        c = tf.expand_dims(c, axis=-1)
        result = tf.where(tf.less_equal(abs_diff, c), abs_diff, (tf.math.pow(diff, 2) + tf.math.pow(c, 2)) / (2 * c))
        return tf.reduce_mean(result, axis=-1)


class HuberLoss(LossFunctionWrapper):
    """
    Huber loss.
    """

    def __init__(self, c_param=0.2):
        super(HuberLoss, self).__init__(fn=self.loss, name="huber_loss")
        self.c_param = c_param

    def loss(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        y_true = tf.reshape(y_true, (-1, tf.reduce_prod(tf.shape(y_true)[1:])))
        y_pred = tf.reshape(y_pred, (-1, tf.reduce_prod(tf.shape(y_pred)[1:])))

        diff = y_pred - y_true
        abs_diff = tf.math.abs(diff)
        c = self.c_param * tf.math.reduce_max(abs_diff, axis=[-1])
        c = tf.expand_dims(c, axis=-1)
        result = tf.where(tf.greater_equal(abs_diff, c), abs_diff, (tf.math.pow(diff, 2) + tf.math.pow(c, 2)) / (2 * c))
        return tf.reduce_mean(result, axis=-1)


class RuberLoss(LossFunctionWrapper):
    """
    Ruber loss.
    """

    def __init__(self, c_param=0.2):
        super(RuberLoss, self).__init__(fn=self.loss, name="ruber_loss")
        self.c_param = c_param

    def loss(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        y_true = tf.reshape(y_true, (-1, tf.reduce_prod(tf.shape(y_true)[1:])))
        y_pred = tf.reshape(y_pred, (-1, tf.reduce_prod(tf.shape(y_pred)[1:])))

        diff = y_pred - y_true
        abs_diff = tf.math.abs(diff)
        c = self.c_param * tf.math.reduce_max(abs_diff, axis=[-1])
        c = tf.expand_dims(c, axis=-1)
        result = tf.where(tf.less_equal(abs_diff, c), abs_diff, tf.math.sqrt(2 * c * abs_diff - tf.math.pow(c, 2)))
        return tf.reduce_mean(result, axis=-1)
