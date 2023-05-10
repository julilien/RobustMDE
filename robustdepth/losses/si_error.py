import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
import tensorflow.keras.backend as keras_backend


class SIError(LossFunctionWrapper):
    """
    As described by Eigen et al., 2014
    """
    def __init__(self, lambda_val=0.5, loss_fn=None, name="si_error"):
        if loss_fn is None:
            loss_fn = self.loss
        super(SIError, self).__init__(fn=loss_fn, name=name)
        self.lambda_val = lambda_val

    def loss(self, y_true, y_pred):
        # Assert batch dimension (in case a single item is given)
        if tf.rank(y_true) == 2:
            y_true = tf.expand_dims(y_true, axis=0)
        if tf.rank(y_pred) == 2:
            y_pred = tf.expand_dims(y_pred, axis=0)

        # Output prediction is in log space
        y_pred_log = y_pred
        y_true = keras_backend.clip(y_true, 1e-7, None)

        y_true_log = tf.math.log(y_true)

        log_diff = y_pred_log - y_true_log

        log_diff_square = tf.math.square(log_diff)
        sum1 = tf.reduce_mean(log_diff_square, axis=[1, 2])

        sum2 = tf.math.square(tf.math.reduce_mean(log_diff, axis=[1, 2]))

        result = sum1 - self.lambda_val * sum2
        return result


class ScaledSIError(SIError):
    """
    Scaled version as used by BTS and AdaBins.
    """
    def __init__(self, lambda_val=0.85, alpha_val=10.):
        # Default parameters as used in BTS
        super(ScaledSIError, self).__init__(lambda_val=lambda_val, loss_fn=self.loss, name="scaled_si_error")
        self.lambda_val = lambda_val
        self.alpha_val = alpha_val

    def loss(self, y_true, y_pred):
        return self.alpha_val * tf.math.sqrt(super(ScaledSIError, self).loss(y_true, y_pred))
