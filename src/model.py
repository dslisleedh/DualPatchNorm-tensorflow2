import tensorflow as tf
import tensorflow_addons as tfa

import einops

from src.layers import *


class ViT(tf.keras.models.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.patch_embedding = PatchEmbedding(
                config.patch_size, config.n_filters, config.norm
        )
        self.cls_token = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=(1, 1, config.n_filters), stddev=0.02),
            trainable=True, name='cls_token', dtype=tf.float32
        )
        self.positional_encoding = PositionalEncoding()
        drop_rates = tf.reshape(
            tf.linspace(0., config.drop_rate, config.n_blocks * 2), (config.n_blocks, 2)
        )
        self.feature_extractor = tf.keras.Sequential([
            ViTBlock(config.n_heads, config.expansion_rate, [float(d_r) for d_r in drop_rates[i]])
            for i in range(config.n_blocks)
        ] + [
            tf.keras.layers.Lambda(lambda x: tf.gather(x, 0, axis=1)),
            tf.keras.layers.LayerNormalization()
        ])
        self.classifier = tf.keras.layers.Dense(
            config.n_labels, kernel_initializer=tf.keras.initializers.zeros, activation='softmax'
        )

    @tf.function(jit_compile=True)
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        patch = self.patch_embedding(inputs)
        feats = self.positional_encoding(
            tf.concat(
                [tf.broadcast_to(self.cls_token, (tf.shape(inputs)[0],) + (1,) + (self.config.n_filters,)), patch],
                axis=1
            )
        )
        feats = self.feature_extractor(feats)
        y_hat = self.classifier(feats)
        return y_hat
