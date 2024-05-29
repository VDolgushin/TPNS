import numpy as np
from lab3b.layers.Layer import Layer


class ReLU(Layer):
    def __init__(self):
        self.output_data = None

    def calculate_output(self, input_data):
        self.output_data = np.maximum(0, input_data)
        return self.output_data

    def back_propagation(self, do, learning_rate):
        return (self.output_data > 0) * do
