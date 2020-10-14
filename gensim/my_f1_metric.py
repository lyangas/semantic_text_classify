# создаем f1-метрику (по умолчанию она отсутствует в TF)
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras import metrics
from tensorflow.python.keras.utils import metrics_utils


class F1Score(metrics.Metric):
    """Computes the f1_score of the predictions with respect to the labels.

    For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [1, 0, 1, 1]
    then the f1_score value is 0.66. If the weights were specified as
    [0, 0, 1, 0] then the precision value would be 1.

    The metric creates three local variables, `true_positives`, `false_positives` and
    `false_negatives` that are used to compute the f1 score.
    This value is ultimately returned as `f1_score`, an operation that
    simply computes 2* `precision`*`recall` /(`precision` + `recall`).
    Where `precision` is ...
    and `recall` is ...

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage: ...
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        """Creates a `F1Score` instance.

        Args:
          thresholds: (Optional) Defaults to 0.5. A float value or a python
            list/tuple of float threshold values in [0, 1]. A threshold is compared
            with prediction values to determine the truth value of predictions
            (i.e., above the threshold is `true`, below is `false`). One metric
            value is generated for each threshold value.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(F1Score, self).__init__(name=name, dtype=dtype)
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=0.5)
        self.tp = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=metrics.init_ops.zeros_initializer)
        self.fp = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=metrics.init_ops.zeros_initializer)
        self.fn = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=metrics.init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive, false positive and false negative statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        metrics_utils.update_confusion_matrix_variables({metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.tp,
                                                         metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.fp,
                                                         metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.fn
                                                         }, y_true, y_pred, self.thresholds,
                                                        sample_weight=sample_weight)

    def result(self):
        precision = math_ops.div_no_nan(self.tp, self.tp + self.fp)
        recall = math_ops.div_no_nan(self.tp, self.tp + self.fn)
        numerator = math_ops.multiply(precision, recall)
        denominator = math_ops.add(precision, recall)
        frac = math_ops.div_no_nan(numerator, denominator)
        result = math_ops.multiply(tf.constant(2.), frac)

        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        for v in self.variables:
            K.set_value(v, np.zeros((num_thresholds,)))
