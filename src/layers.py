import tensorflow as tf
import tensorflow_addons as tfa

import einops
from einops.layers.keras import Rearrange
import numpy as np

from typing import Sequence, Optional


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(
            self, patch_size: Sequence[int], n_filters: int, norm: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.n_filters = n_filters
        assert norm in [None, 'pre', 'post', 'dual']
        self.norm = norm

        self.forward = tf.keras.Sequential()
        self.forward.add(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1])
        )
        if norm in ['pre', 'dual']:
            self.forward.add(tf.keras.layers.LayerNormalization())
        self.forward.add(tf.keras.layers.Dense(n_filters))
        if norm in ['post', 'dual']:
            self.forward.add(tf.keras.layers.LayerNormalization())

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.forward(inputs)


class PositionalEncoding(tf.keras.layers.Layer):
    def build(self, input_shape):
        pe = np.array(
            [[(pos / 10000 ** (2 * (i // 2) / input_shape[-1])) for i in range(input_shape[-1])]
             for pos in range(input_shape[-2])],
            dtype=np.float32
        )
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        self.pe = tf.Variable(pe[np.newaxis, :], trainable=False, dtype=tf.float32, name="positional_encoding")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return inputs + tf.broadcast_to(self.pe, (tf.shape(inputs)[0], *self.pe.shape[1:]))


class MHSA(tf.keras.layers.Layer):
    def __init__(self, n_heads: int = 8):
        super(MHSA, self).__init__()
        self.n_heads = n_heads

    def build(self, input_shape):
        self.ln = tf.keras.layers.LayerNormalization()
        self.to_qkv = tf.keras.layers.Dense(input_shape[-1] * 3, use_bias=False)
        self.to_out = tf.keras.layers.Dense(input_shape[-1], use_bias=False)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        qkv = self.to_qkv(self.ln(inputs))
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = einops.rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)
        k = einops.rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)
        v = einops.rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)

        attn = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)

        out = einops.rearrange(out, 'b h t d -> b t (h d)')
        out = self.to_out(out)
        return out


class MLP(tf.keras.layers.Layer):
    def __init__(self, expansion_rate: int = 4):
        super(MLP, self).__init__()
        self.expansion_rate = expansion_rate

    def build(self, input_shape):
        self.ln = tf.keras.layers.LayerNormalization()
        self.mlp1 = tf.keras.layers.Dense(input_shape[-1] * self.expansion_rate, activation='gelu')
        self.mlp2 = tf.keras.layers.Dense(input_shape[-1])

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.ln(inputs)
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x


class ViTBlock(tf.keras.layers.Layer):
    def __init__(self, n_heads: int, expansion_rate: int, drop_rate: Sequence[float]):
        super(ViTBlock, self).__init__()
        self.n_heads = n_heads
        self.expansion_rate = expansion_rate
        self.drop_rate = drop_rate

        self.spatial_drop = tfa.layers.StochasticDepth(1. - drop_rate[0])
        self.mhsa = MHSA(n_heads)
        self.channel_drop = tfa.layers.StochasticDepth(1. - drop_rate[1])
        self.mlp = MLP(expansion_rate)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.spatial_drop([inputs, self.mhsa(inputs)])
        x = self.channel_drop([x, self.mlp(x)])
        return x
