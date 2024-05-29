from lab3b.model.CrossEntropy import CrossEntropy
from lab3b.layers import Layer
import numpy as np


class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.loss_func = CrossEntropy()

    def calculate_output(self, input_data: np.ndarray):
        for layer in self.layers:
            input_data = layer.calculate_output(input_data)
        return input_data

    def back_propagation(self, d_output: np.ndarray, learning_rate) -> None:
        for layer in reversed(self.layers):
            d_output = layer.back_propagation(d_output, learning_rate=learning_rate)

    def train(self, x_train, y_train, epochs, learning_rate, batch_size=1):
        data_size = len(x_train)
        for epoch in range(epochs):
            indices = np.random.permutation(data_size)
            x_train_shuffled = x_train#[indices]
            y_train_shuffled = y_train#[indices]
            loss_train = 0
            for batch_start in range(0, data_size, batch_size):
                batch_end = min(batch_start + batch_size, data_size)
                x_batch = x_train_shuffled[batch_start:batch_end]
                y_batch = y_train_shuffled[batch_start:batch_end]
                predictions_train = self.calculate_output(x_batch)
                loss_train += self.loss_func(y_batch, predictions_train)
                d_output_train = self.loss_func.gradient(y_batch, predictions_train)
                self.back_propagation(d_output_train, learning_rate)
            print("Epoch:", epoch + 1,  "Loss:", f"{loss_train / data_size :.4f}")

    def predict(self, x_test):
        return self.calculate_output(x_test)
