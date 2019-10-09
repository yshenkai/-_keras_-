import keras.backend as K
from keras.engine.base_layer import Layer
from keras import initializers
import tensorflow as tf
class Spatial_Attention(Layer):

    def build(self, input_shape):
        _,self.H,self.W,self.C=input_shape
        self.w=tf.get_variable("SpatialAttention_w_s", [self.C, 1],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal)

        self.b=tf.get_variable("SpatialAttention_b_s", [1],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        self.trainable_weights=[self.w,self.b]
        super(Spatial_Attention,self).build(input_shape)
    def call(self, inputs, **kwargs):
        spatial_attention_fm = tf.matmul(tf.reshape(inputs, [-1, self.C]), self.w) + self.b
        spatial_attention_fm = tf.nn.sigmoid(tf.reshape(spatial_attention_fm, [-1, self.W * self.H]))
        #         spatial_attention_fm = tf.clip_by_value(tf.nn.relu(tf.reshape(spatial_attention_fm,
        #                                                                       [-1, W * H])),
        #                                                 clip_value_min = 0,
        #                                                 clip_value_max = 1)
        attention = tf.reshape(tf.concat([spatial_attention_fm] * self.C, axis=1), [-1, self.H, self.W, self.C])
        attended_fm = attention * inputs
        return attended_fm
    def compute_output_shape(self, input_shape):
        return input_shape

