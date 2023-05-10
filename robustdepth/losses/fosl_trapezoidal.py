import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
import abc


class FOSLTrapezoidal(LossFunctionWrapper):
    def __init__(self, name, depth_factor, core_fns, support_fns):
        super(FOSLTrapezoidal, self).__init__(fn=self.loss, name=name)

        self.core_fns = core_fns
        self.support_fns = support_fns
        self.depth_factor = depth_factor

    @abc.abstractmethod
    def loss(self, y_true, y_pred):
        pass

    def is_unified_core(self):
        return len(self.core_fns) == 1

    def is_unified_support(self):
        return len(self.support_fns) == 1

    def induce_cores(self, values):
        left_core = values - self.core_fns[0](values)
        if self.is_unified_core():
            right_core = values + self.core_fns[0](values)
        else:
            right_core = values + self.core_fns[1](values)

        return left_core, right_core

    def induce_supports(self, values, left_core, right_core):
        left_support = left_core - self.support_fns[0](values)
        if self.is_unified_support():
            right_support = right_core + self.support_fns[0](values)
        else:
            right_support = right_core + self.support_fns[1](values)
        return left_support, right_support

    @staticmethod
    def assert_batch_dimension(values):
        if tf.rank(values) == 2:
            values = tf.expand_dims(values, axis=0)
        return values


class FOSLL1Trapezoidal(FOSLTrapezoidal):
    def __init__(self, depth_factor, core_fns, support_fns, name=None):
        if name is None:
            name = "fosll1_trapezoidal_loss"
        super(FOSLL1Trapezoidal, self).__init__(name=name, depth_factor=depth_factor, core_fns=core_fns,
                                                support_fns=support_fns)

    def loss(self, y_true, y_pred):
        y_true = self.depth_factor * y_true
        y_pred = self.depth_factor * y_pred

        # Assert batch dimension (in case a single item is given)
        if tf.rank(y_true) == 2:
            y_true = tf.expand_dims(y_true, axis=0)
        if tf.rank(y_pred) == 2:
            y_pred = tf.expand_dims(y_pred, axis=0)

        left_core, right_core = self.induce_cores(y_true)
        left_support, right_support = self.induce_supports(y_true, left_core, right_core)

        delta_left = left_core - left_support
        delta_right = right_support - right_core

        return tf.where(tf.less(y_pred, left_support),
                        tf.abs(y_pred - left_core) - 0.5 * delta_left,
                        tf.where(tf.less_equal(y_pred, left_core),
                                 0.5 / delta_left * tf.math.pow(y_pred - left_core, 2),
                                 tf.where(tf.less_equal(y_pred, right_core), 0.,
                                          tf.where(tf.less_equal(y_pred, right_support),
                                                   0.5 / delta_right * tf.math.pow(
                                                       y_pred - right_core, 2),
                                                   tf.abs(
                                                       y_pred - right_core) - 0.5 * delta_right))))
