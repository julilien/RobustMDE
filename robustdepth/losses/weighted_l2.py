import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper, MeanSquaredError


class WeightedL2Loss(LossFunctionWrapper):
    """
    Weighted least squares.
    """

    def __init__(self, loss_fn=None, name="weighted_l2"):
        if loss_fn is None:
            loss_fn = self.loss
        super(WeightedL2Loss, self).__init__(fn=loss_fn, name=name)

        self.int_error = MeanSquaredError()

    @staticmethod
    def weighting_function(y_true):
        return 0.0012 + 0.0019 * tf.math.pow(y_true - 0.4, 2)

    def loss(self, y_true, y_pred):
        if tf.rank(y_pred) == 3:
            y_pred = tf.expand_dims(y_pred, axis=-1)
        if tf.rank(y_true) == 3:
            y_true = tf.expand_dims(y_true, axis=-1)

        weights = 1. / tf.math.pow(WeightedL2Loss.weighting_function(y_true), 2)
        return self.int_error(y_true, y_pred, sample_weight=weights)
