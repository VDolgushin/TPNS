from lab3b.layers.Layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def calculate_output(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def back_propagation(self, d_output: np.ndarray, **kwargs) -> np.ndarray:
        return d_output.reshape(self.input_shape)