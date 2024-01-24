# Custom L1 Distance Layer Module

import tensorflow as tf
from tensorflow.keras.layers import Layer


# Custom L1 Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


model = tf.keras.models.load_model('siamesemodel.h5',
                                   custom_objects={'L1Dist': L1Dist})

