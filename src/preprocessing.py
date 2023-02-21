import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Sequence

from functools import partial


def load_data(batch_size: int) -> Sequence[tf.data.Dataset]:
    dataset_name = 'imagenet_resized/64x64'
    # dataset_name = 'cifar10'
    image_size = 64

    train_ds, valid_ds, test_ds = tfds.load(
        dataset_name, as_supervised=True, split=['train[:85%]', 'train[85%:]', 'validation'],
        shuffle_files=False
    )
    augment_func = partial(preprocessing, image_size=image_size, augment=True)
    non_augment_func = partial(preprocessing, image_size=image_size, augment=False)
    train_ds = train_ds.map(augment_func, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000) \
        .batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(non_augment_func, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(non_augment_func, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds


def preprocessing(x, y, image_size: int, augment: bool = False):
    x = tf.cast(x, tf.float32) / 255.

    if augment:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        crop_size = tf.random.uniform(shape=[], minval=image_size - 4, maxval=image_size + 1, dtype=tf.int32)
        x = tf.image.random_crop(x, size=(crop_size, crop_size, 3))
        if crop_size != image_size:
            x = tf.image.resize(x, size=(image_size, image_size))
    x = tf.ensure_shape(x, shape=(image_size, image_size, 3))
    return x, y
