from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as mp
import seaborn
import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

def clear_data(df):
    df['stalk-root'] = df['stalk-root'].fillna(df['stalk-root'].mode()[0])
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop('veil-type', axis=1)
    return df

class Neuron:
    output = None
    inputs = None
    weights = None
    delta = None

    def __init__(self, inputs_n):
        self.weights = [abs(np.random.default_rng().random()/100) for i in range(inputs_n)]

    def activate(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs)
        return self.logistic_function(self.output)

    def logistic_function(self, x):
        return 1/(1 + np.exp(-x))

    def derivative_of_logistic_function(self, x):
        return self.logistic_function(x) * (1 - self.logistic_function(x))

class MultyLayerPerceptronClassifier:
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
        return round(self.output_layer.activate(layer_outputs))

    def back_propagation(self, inputs, correct_output, nu):
        output = self.predict(inputs)
        error = output - correct_output
        self.loss_curve.append((correct_output - output) ** 2)

        self.output_layer.delta = error * self.output_layer.derivative_of_logistic_function(self.output_layer.output)

        for j, neuron in enumerate(self.hidden_layers[-1]):
            neuron.delta = neuron.derivative_of_logistic_function(neuron.output) * self.output_layer.delta * \
                           self.output_layer.weights[j]

        for i in range(len(self.hidden_layers) - 2, -1, -1):
            for j, neuron in enumerate(self.hidden_layers[i]):
                neuron.delta = neuron.derivative_of_logistic_function(neuron.output) * np.dot(
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


def classifier_mushrooms():
    mushroom = fetch_ucirepo(id=73)
    df = mushroom.data.original
    print(df['veil-type'].unique())
    df = clear_data(df)
    df = df.apply(lambda x: pd.factorize(x)[0])
    x = df.drop('poisonous', axis=1)
    y = df['poisonous']
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)

    mlp_class = MultyLayerPerceptronClassifier(3, x.shape[1], [50])
    mlp_class.fit(trainX.values, trainY, 1)
    y_pred = []
    for i in testX.values:
        y_pred.append(mlp_class.predict(i))
    df_pred_res = pd.DataFrame({'Actual': testY, 'Predicted': y_pred, 'Diff': testY - y_pred})
    print(df_pred_res.to_string())
    print(100 - sum(abs(df_pred_res['Diff'])) / len(df_pred_res['Diff']) * 100, "%")
    #pd.DataFrame(mlp_class.loss_curve_).plot(figsize=(8, 5))
    #mp.show()


def main():
    classifier_mushrooms()


if __name__ == '__main__':
    main()
