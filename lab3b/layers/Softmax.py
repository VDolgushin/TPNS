import numpy as np
from lab3b.layers.Layer import Layer


class Softmax(Layer):
    def calculate_output(self, input_data):
        exp = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def back_propagation(self, d_output, learning_rate):
        return d_output
