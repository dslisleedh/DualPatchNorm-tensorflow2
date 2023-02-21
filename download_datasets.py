import tensorflow as tf
import tensorflow_datasets as tfds

import src


"""
Optuna raises error when downloading imagenet_resized dataset.
So we need to download it and generate samples first.
"""


if __name__ == '__main__':
    train_ds, valid_ds, test_ds = src.load_data(512)

    print("Done !!!")
    print("run train.py to train the model")
