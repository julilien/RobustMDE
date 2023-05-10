import tensorflow as tf
from tensorflow.python.keras.metrics import Mean
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as keras_backend


class ResizedError(Mean):
    def __init__(self, target_shape, model_output_shape, batch_size, error_metric, name, dtype=None):
        super(ResizedError, self).__init__(name, dtype=dtype)
        self.target_shape = target_shape
        self.model_output_shape = model_output_shape

        self.batch_size = batch_size
        self.error_metric = error_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        if tf.rank(y_pred) == 2:
            y_pred = tf.expand_dims(y_pred, axis=0)
        if tf.rank(y_pred) == 3:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        y_pred = tf.ensure_shape(y_pred, [self.batch_size] + self.model_output_shape[:2] + [1])
        y_pred = tf.image.resize(y_pred, self.target_shape)

        return self.error_metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.error_metric.result()


class DistanceCensoredError(Mean):
    def __init__(self, batch_size, max_depth_distance, error_metric, name, dtype=None):
        super(DistanceCensoredError, self).__init__(name, dtype=dtype)
        self.batch_size = batch_size
        self.error_metric = error_metric
        self.max_depth_distance = max_depth_distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [self.batch_size, -1])
        y_pred = tf.reshape(y_pred, [self.batch_size, -1])

        # Targets are assumed to be within [0, self.max_depth_distance]
        depth_mask = tf.logical_and(y_true < self.max_depth_distance, y_true > 1e-3)

        y_true = y_true[depth_mask]
        y_pred = y_pred[depth_mask]

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        # Assure batch dimension before cropping
        if tf.rank(y_true) == 1:
            y_true = tf.expand_dims(y_true, axis=0)
        if tf.rank(y_pred) == 1:
            y_pred = tf.expand_dims(y_pred, axis=0)

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        return self.error_metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.error_metric.result()


class DistanceUncensoredError(Mean):
    def __init__(self, batch_size, error_metric, name, dtype=None):
        super(DistanceUncensoredError, self).__init__(name, dtype=dtype)
        self.batch_size = batch_size
        self.error_metric = error_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [self.batch_size, -1])
        y_pred = tf.reshape(y_pred, [self.batch_size, -1])

        # As commonly done
        depth_mask = y_true > 1e-3
        y_true = y_true[depth_mask]
        y_pred = y_pred[depth_mask]

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        # Assure batch dimension before cropping
        if tf.rank(y_true) == 1:
            y_true = tf.expand_dims(y_true, axis=0)
        if tf.rank(y_pred) == 1:
            y_pred = tf.expand_dims(y_pred, axis=0)

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        return self.error_metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.error_metric.result()


class PreprocessedDepthMetric(Mean):
    def __init__(self, max_depth_factor, log_space=False, name=None, dtype=None, rescale_target=False,
                 upper_bound_pred=True):
        super(PreprocessedDepthMetric, self).__init__(name, dtype=dtype)
        self.max_depth_factor = max_depth_factor
        self.log_space = log_space
        self.rescale_target = rescale_target
        self.upper_bound_pred = upper_bound_pred

    def preprocess_input(self, y_true, y_pred):
        if self.log_space:
            y_pred = math_ops.exp(y_pred)

        y_pred = y_pred * self.max_depth_factor
        if self.rescale_target:
            y_true = y_true * self.max_depth_factor

        if self.upper_bound_pred:
            upper_bound = self.max_depth_factor
        else:
            upper_bound = None

        y_pred = keras_backend.clip(y_pred, 1e-3, upper_bound)

        if tf.shape(y_true)[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        if tf.shape(y_pred)[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)

        return y_true, y_pred


class CroppingError(Mean):
    def __init__(self, cropping_coords, error_metric, name, dtype=None):
        super(CroppingError, self).__init__(name, dtype=dtype)
        self.error_metric = error_metric
        self.cropping_coords = cropping_coords

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        # Assure batch dimension before cropping
        if tf.rank(y_true) == 2:
            y_true = tf.expand_dims(y_true, axis=0)
        if tf.rank(y_pred) == 2:
            y_pred = tf.expand_dims(y_pred, axis=0)

        # Do cropping
        y_true = y_true[:, self.cropping_coords[0]: self.cropping_coords[1],
                 self.cropping_coords[2]:self.cropping_coords[3]]
        y_pred = y_pred[:, self.cropping_coords[0]: self.cropping_coords[1],
                 self.cropping_coords[2]:self.cropping_coords[3]]

        return self.error_metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.error_metric.result()


class RootMeanSquaredError(PreprocessedDepthMetric):
    def __init__(self, max_depth_factor, log_space=False, name='rmse', dtype=None, rescale_target=False,
                 upper_bound_pred=True):
        super(RootMeanSquaredError, self).__init__(max_depth_factor, log_space, name, dtype=dtype,
                                                   rescale_target=rescale_target, upper_bound_pred=upper_bound_pred)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = super(RootMeanSquaredError, self).preprocess_input(y_true, y_pred)

        error_sq = math_ops.sqrt(tf.math.reduce_mean(tf.pow(y_true - y_pred, 2), axis=-1))
        return super(RootMeanSquaredError, self).update_state(error_sq, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class AbsRel(PreprocessedDepthMetric):
    def __init__(self, max_depth_factor, log_space=False, name='abs_rel', dtype=None, rescale_target=False,
                 upper_bound_pred=True):
        super(AbsRel, self).__init__(max_depth_factor, log_space, name, dtype=dtype, rescale_target=rescale_target,
                                     upper_bound_pred=upper_bound_pred)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = super(AbsRel, self).preprocess_input(y_true, y_pred)

        error = tf.math.reduce_mean(tf.math.abs(y_pred - y_true) /
                                    keras_backend.clip(y_true, keras_backend.epsilon(), None), axis=-1)

        return super(AbsRel, self).update_state(error, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class SqRel(PreprocessedDepthMetric):
    def __init__(self, max_depth_factor, log_space=False, name='sq_rel', dtype=None, rescale_target=False,
                 upper_bound_pred=True):
        super(SqRel, self).__init__(max_depth_factor, log_space, name, dtype=dtype, rescale_target=rescale_target,
                                    upper_bound_pred=upper_bound_pred)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = super(SqRel, self).preprocess_input(y_true, y_pred)

        error = tf.math.reduce_mean(
            tf.pow(y_true - y_pred, 2) / keras_backend.clip(y_true, keras_backend.epsilon(), None),
            axis=-1)

        return super(SqRel, self).update_state(error, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class Log10(PreprocessedDepthMetric):
    def __init__(self, max_depth_factor, log_space=False, name='log_10', dtype=None, rescale_target=False,
                 upper_bound_pred=True):
        super(Log10, self).__init__(max_depth_factor, log_space, name, dtype=dtype, rescale_target=rescale_target,
                                    upper_bound_pred=upper_bound_pred)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = super(Log10, self).preprocess_input(y_true, y_pred)

        def log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        error = tf.math.reduce_mean(tf.math.abs(log10(y_pred) - log10(y_true)), axis=-1)

        return super(Log10, self).update_state(error, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class RMSELog(PreprocessedDepthMetric):
    def __init__(self, max_depth_factor, log_space=False, name='rmse_log', dtype=None, rescale_target=False,
                 upper_bound_pred=True):
        super(RMSELog, self).__init__(max_depth_factor, log_space, name, dtype=dtype, rescale_target=rescale_target,
                                      upper_bound_pred=upper_bound_pred)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = super(RMSELog, self).preprocess_input(y_true, y_pred)

        error = math_ops.sqrt(
            tf.math.reduce_mean(tf.pow(tf.math.log(y_pred) - tf.math.log(y_true), 2), axis=-1))

        return super(RMSELog, self).update_state(error, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class DeltaAccuracy(PreprocessedDepthMetric):
    def __init__(self, delta, max_depth_factor, log_space=False, name="delta_acc", dtype=None, rescale_target=False,
                 upper_bound_pred=True):
        super(DeltaAccuracy, self).__init__(max_depth_factor, log_space, name, dtype=dtype,
                                            rescale_target=rescale_target, upper_bound_pred=upper_bound_pred)
        self.delta = delta

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = super(DeltaAccuracy, self).preprocess_input(y_true, y_pred)

        max_vals = tf.maximum(y_true / y_pred, y_pred / y_true)

        delta_acc = tf.math.reduce_mean(tf.cast(tf.less(max_vals, self.delta), dtype=tf.float32), axis=-1)

        return super(DeltaAccuracy, self).update_state(delta_acc, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class RMSEFOSL(Mean):
    """
    Computes root mean squared error metric between `y_true` and `y_pred` for prediction given in log-space.
    Thus, exp() is applied to the prediction before calculating the RMSE metric.
    """

    def __init__(self, name='root_mean_squared_error', dtype=None):
        super(RMSEFOSL, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        y_pred = tf.squeeze(y_pred)
        y_true = tf.squeeze(y_true[0])

        if tf.rank(y_true) == 1:
            y_true = tf.expand_dims(y_true, axis=0)
        if tf.rank(y_pred) == 1:
            y_pred = tf.expand_dims(y_pred, axis=0)

        error_sq = math_ops.sqrt(tf.reduce_mean(tf.pow(y_true - y_pred, 2), axis=-1))
        return super(RMSEFOSL, self).update_state(error_sq, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)
