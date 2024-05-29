from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as mp
import seaborn
import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    output = None
    inputs = None
    weights = None
    delta = None

    def __init__(self, inputs_n):
        self.weights = abs(np.random.randn(inputs_n) * np.sqrt(2. / inputs_n))

    def activate(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs)
        return self.relu(self.output)

    def relu(self, x):
        return max(x, 0)

    def derivative_of_relu(self, x):
        return 0 if x <= 0 else 1


class MultyLayerPerceptronRegressor:
    hidden_layers = None
    output_layer = None
    epochs = None
    loss_curve = None

    def __init__(self, epochs, input_layer_size, layers_sizes=[]):
        self.hidden_layers = [[]]
        self.hidden_layers[0] = [Neuron(input_layer_size) for _ in range(layers_sizes[0])]
        for i in range(1, len(layers_sizes)):
            self.hidden_layers.append([Neuron(layers_sizes[i - 1]) for _ in range(layers_sizes[i])])
        self.output_layer = Neuron(layers_sizes[-1])
        self.epochs = epochs
        self.loss_curve = []

    def predict(self, input_layer):
        layer_outputs = [neuron.activate(input_layer) for neuron in self.hidden_layers[0]]
        for i in range(1, len(self.hidden_layers)):
            layer_outputs = [neuron.activate(layer_outputs) for neuron in self.hidden_layers[i]]
        return self.output_layer.activate(layer_outputs)

    def back_propagation(self, inputs, correct_output, nu):
        output = self.predict(inputs)
        error = -2 * (correct_output - output)
        self.loss_curve.append((correct_output - output) ** 2)

        self.output_layer.delta = error * self.output_layer.derivative_of_relu(self.output_layer.output)

        for j, neuron in enumerate(self.hidden_layers[-1]):
            neuron.delta = neuron.derivative_of_relu(neuron.output) * self.output_layer.delta * \
                           self.output_layer.weights[j]

        for i in range(len(self.hidden_layers) - 2, -1, -1):
            for j, neuron in enumerate(self.hidden_layers[i]):
                neuron.delta = neuron.derivative_of_relu(neuron.output) * np.dot(
                    [n.delta for n in self.hidden_layers[i + 1]],
                    [n.weights[j] for n in self.hidden_layers[i + 1]])

        for i in range(len(self.hidden_layers)):
            for j, neuron in enumerate(self.hidden_layers[i]):
                for k in range(len(neuron.weights)):
                    neuron.weights[k] = neuron.weights[k] - nu * neuron.delta * neuron.inputs[k]

        for k in range(len(self.output_layer.weights)):
            self.output_layer.weights[k] = self.output_layer.weights[k] - nu * self.output_layer.delta * \
                                           self.output_layer.inputs[k]

    def fit(self, inputs, outputs, learning_rate):
        for j in range(self.epochs):
            for i in range(len(outputs)):
                self.back_propagation(inputs[i], outputs.iloc[i], learning_rate)
            print("epoch: ", j)



def regressor_laptop():
    data = pd.read_csv('data/Laptop_price.csv')
    df = pd.DataFrame(data)
    df['Brand'] = pd.factorize(df['Brand'])[0]
    x = df.drop('Price', axis=1)
    y = df['Price']
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)
    mlp_reg = MultyLayerPerceptronRegressor(10, x.shape[1], [50])
    mlp_reg.fit(trainX.values, trainY, 1e-8)
    y_pred = []
    for i in testX.values:
        y_pred.append(mlp_reg.predict(i))
    df_pred_res = pd.DataFrame({'Actual': testY, 'Predicted': y_pred, 'Diff': testY - y_pred})
    print(df_pred_res.to_string())
    print(sum(abs(df_pred_res['Diff'])) / len(df_pred_res['Diff']))
    #print(100 - sum(abs((df_pred_res['Diff'] / df_pred_res['Actual']) * 100)) / len(df_pred_res['Diff']), "%")



def main():
    regressor_laptop()


if __name__ == '__main__':
    main()
