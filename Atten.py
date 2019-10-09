from keras.engine.topology import Layer
import tensorflow as tf
class Atten(Layer):
    def build(self, input_shape):
        _,self.H,self.W,self.C=input_shape
        self.w_c = tf.get_variable("ChannelWiseAttention_w_c", [self.C, self.C],
                                 dtype=tf.float32,
                                 initializer=tf.initializers.orthogonal)

        self.b_c = tf.get_variable("ChannelWiseAttention_b_c", [self.C],
                                 dtype=tf.float32,
                                 initializer=tf.initializers.zeros)
        self.w_s = tf.get_variable("SpatialAttention_w_s", [self.C, 1],
                                 dtype=tf.float32,
                                 initializer=tf.initializers.orthogonal)

        self.b_s = tf.get_variable("SpatialAttention_b_s", [1],
                                 dtype=tf.float32,
                                 initializer=tf.initializers.zeros)
        self.trainable_weights=[self.w_c,self.w_s,self.b_s,self.b_c]
        super(Atten,self).build(input_shape)
    def call(self, inputs, **kwargs):
        transpose_feature_map = tf.transpose(tf.reduce_mean(inputs, [1, 2], keepdims=True),
                                             perm=[0, 3, 1, 2])
        channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map,
                                                         [-1, self.C]), self.w_c) + self.b_c
        channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)

        attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (self.H * self.W),
                                         axis=1), [-1, self.H, self.W, self.C])
        inputs = attention * inputs

        spatial_attention_fm = tf.matmul(tf.reshape(inputs, [-1, self.C]), self.w_s) + self.b_s
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


