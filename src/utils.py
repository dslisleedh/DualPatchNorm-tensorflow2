import tensorflow as tf
import tensorflow_addons as tfa


class SparseF1Score(tfa.metrics.F1Score):
    def __init__(self, *args, **kwargs):
        super(SparseF1Score, self).__init__(*args, **kwargs)
        self._num_classes = kwargs['num_classes']

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, self._num_classes)
        y_true = tf.gather(y_true, 0, axis=1)
        return super(SparseF1Score, self).update_state(y_true, y_pred, sample_weight)
