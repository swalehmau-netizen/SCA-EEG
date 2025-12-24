import tensorflow as tf
from tensorflow.keras.layers import Layer

class ScaledCustomAttention(Layer):
    """
    Scaled Custom Attention (SCA) layer for EEG time-series modeling.
    Applies timestep-wise feature weighting with dimension-based scaling.
    """

    def __init__(self, **kwargs):
        super(ScaledCustomAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, T, d)
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1],),
            initializer="normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1],),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch_size, T, d)
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)

        # scale by d^2
        d = tf.cast(tf.shape(x)[-1], tf.float32)
        e_scaled = e / tf.math.square(d)

        # attention weights
        alpha = tf.keras.activations.softmax(e_scaled, axis=1)

        # weighted aggregation
        output = tf.reduce_sum(x * tf.expand_dims(alpha, -1), axis=1)
        return output
