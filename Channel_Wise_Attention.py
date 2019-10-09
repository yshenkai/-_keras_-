from keras.engine.base_layer import Layer
from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
import tensorflow as tf
class Channel_Wise_Attention(Layer):
    def build(self, input_shape):
        _,self.H,self.W,self.C=input_shape

        self.w=tf.get_variable("ChannelWiseAttention_w_s", [self.C, self.C],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal)

        self.b=tf.get_variable("ChannelWiseAttention_b_s", [self.C],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        self.trainable_weights=[self.w,self.b]
        super(Channel_Wise_Attention,self).build(input_shape)
    def call(self, inputs, **kwargs):
        transpose_feature_map = tf.transpose(tf.reduce_mean(inputs, [1, 2], keepdims=True),
                                             perm=[0, 3, 1, 2])
        channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map,
                                                         [-1, self.C]),self.w) + self.b
        channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)

        attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (self.H * self.W),
                                         axis=1), [-1, self.H, self.W, self.C])
        attended_fm = attention * inputs
        return attended_fm

