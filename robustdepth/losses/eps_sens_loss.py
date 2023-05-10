import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


class EpsSensL1Loss(LossFunctionWrapper):
    def __init__(self, batch_size, eps_param=1., loss_fn=None, name="eps_sens_mae"):
        if loss_fn is None:
            loss_fn = self.loss
        super(EpsSensL1Loss, self).__init__(fn=loss_fn, name=name)
        self.eps_param = eps_param
        self.batch_size = batch_size

    def eps_function(self, y_true):
        return self.eps_param

    def loss(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [self.batch_size, -1])
        y_pred = tf.reshape(y_pred, [self.batch_size, -1])

        outer_left = y_true - self.eps_function(y_true)
        outer_right = y_true + self.eps_function(y_true)

        result = tf.where(tf.less(y_pred, outer_left), tf.math.abs(y_pred - outer_left),
                          tf.where(tf.less(y_pred, outer_right), tf.zeros_like(y_pred),
                                   tf.math.abs(y_pred - outer_right)))

        if tf.rank(result) > 1:
            return tf.reduce_mean(result, axis=-1)
        else:
            return result


class EpsSensL2Loss(LossFunctionWrapper):
    def __init__(self, batch_size, eps_param=1., loss_fn=None, name="eps_sens_mse"):
        if loss_fn is None:
            loss_fn = self.loss
        super(EpsSensL2Loss, self).__init__(fn=loss_fn, name=name)
        self.eps_param = eps_param
        self.batch_size = batch_size

    def eps_function(self, y_true):
        return self.eps_param

    def loss(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [self.batch_size, -1])
        y_pred = tf.reshape(y_pred, [self.batch_size, -1])

        outer_left = y_true - self.eps_function(y_true)
        outer_right = y_true + self.eps_function(y_true)

        result = tf.where(tf.less(y_pred, outer_left), tf.math.pow(y_pred - outer_left, 2),
                          tf.where(tf.less(y_pred, outer_right), tf.zeros_like(y_pred),
                                   tf.math.pow(y_pred - outer_right, 2)))

        if tf.rank(result) > 1:
            return tf.reduce_mean(result, axis=-1)
        else:
            return result
