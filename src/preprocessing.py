import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Sequence

from functools import partial


def load_data(batch_size: int) -> Sequence[tf.data.Dataset]:
    train_ds, valid_ds, test_ds = tfds.load(
        'cifar10', as_supervised=True, split=['train[:80%]', 'train[80%:]', 'test']
    )
    augment_func = partial(preprocessing, augment=True)
    non_augment_func = partial(preprocessing, augment=False)
    train_ds = train_ds.map(augment_func, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000) \
        .batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(non_augment_func, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(non_augment_func, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds


def preprocessing(x, y, augment: bool = False):
    x = tf.cast(x, tf.float32) / 255.

    if augment:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        crop_size = tf.random.uniform(shape=[], minval=28, maxval=32, dtype=tf.int32)
        x = tf.image.random_crop(x, size=(crop_size, crop_size, 3))
        x = tf.image.resize(x, size=(32, 32))
        x = tf.image.random_brightness(x, max_delta=0.5)

    return x, y
