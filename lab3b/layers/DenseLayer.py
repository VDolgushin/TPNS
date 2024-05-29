import numpy as np
from lab3b.layers.Layer import Layer


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, wScale=0.01):
        self.wScale = wScale
        self.x = None
        self.weights = None
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * self.wScale
        self.bias = np.zeros((1, output_size))

    def calculate_output(self, x):
        self.x = x
        output_data = np.dot(x, self.weights) + self.bias
        return output_data

    def back_propagation(self, d_out, learning_rate):
        dx = np.dot(d_out, self.weights.T)
        dw = np.dot(self.x.T, d_out)
        db = np.sum(d_out, axis=0)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        return dx