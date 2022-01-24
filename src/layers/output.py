import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, GlobalMaxPool2D


class CnnLayer(Layer):
    def __init__(self, filter_nums, kernel_size):
        super(CnnLayer, self).__init__()
        self.kernel_size = kernel_size
        self.filter_nums = filter_nums

        self.cnn_layer = Conv2D(self.filter_nums, self.kernel_size, activation='relu')
        self.pool_layer = GlobalMaxPool2D()

    def call(self, hidden_states):
        hs = tf.expand_dims(hidden_states, axis=-1)
        features = self.cnn_layer(hs)
        outputs = self.pool_layer(features)

        return outputs, features
    
    def get_config(self):
        config = super(CnnLayer, self).get_config()
        config.update({
            'filter_nums': self.filter_nums,
            'kernel_size': self.kernel_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)