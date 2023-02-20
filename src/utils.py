import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np


class SparseF1Score(tfa.metrics.F1Score):
    def __init__(self, *args, **kwargs):
        super(SparseF1Score, self).__init__(*args, **kwargs)
        self._num_classes = kwargs['num_classes']

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, self._num_classes)
        y_true = tf.gather(y_true, 0, axis=1)
        return super(SparseF1Score, self).update_state(y_true, y_pred, sample_weight)


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self, learning_rate: float, warmup_steps: int, total_steps: int, alpha: float = 0.0,
    ):
        self._learning_rate = learning_rate
        self._warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
        self._total_steps = tf.constant(total_steps, dtype=tf.float32)
        self._decay_steps = tf.constant(total_steps - warmup_steps, dtype=tf.float32)
        self._alpha = alpha

        self.cosine_decay = tf.keras.experimental.CosineDecay(
            initial_learning_rate=self._learning_rate,
            decay_steps=self._decay_steps,
            alpha=self._alpha,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.cond(
            step < self._warmup_steps,
            lambda: self._learning_rate * ((step + 1.) / self._warmup_steps),
            lambda: self.cosine_decay(step - self._warmup_steps),
        )
        return lr
