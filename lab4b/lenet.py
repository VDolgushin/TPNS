import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from lab3b.model.Model import Model
from lab3b.layers.DenseLayer import DenseLayer
from lab3b.layers.Softmax import Softmax
from lab4b.AvgPool2DLayer import AvgPool2DLayer
from lab4b.Conv2DLayer import Conv2DLayer
from lab4b.Flatten import Flatten
from tensorflow.keras import datasets
import tensorflow as tf
import keras


def main():

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    model = Model([Conv2DLayer(input_size=1, output_size=6, stride=1, kernel_size=5, padding=2, wScale=0.05),
                   AvgPool2DLayer(pool_size=2, stride=2),
                   Conv2DLayer(input_size=6, output_size=16, stride=1, kernel_size=5, wScale=0.05),
                   AvgPool2DLayer(pool_size=2, stride=2),
                   Conv2DLayer(input_size=16, output_size=120, stride=1, kernel_size=5, wScale=0.05),
                   Flatten(),
                   DenseLayer(input_size=120, output_size=84),
                   DenseLayer(input_size=84, output_size=10),
                   Softmax()
                   ])
    model.train(X_train, y_train, epochs=3, batch_size=100, learning_rate=0.001)
    y_pred = model.predict(X_test)
    s = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i].argmax() == y_test[i].argmax():
            s += 1
    print(s/y_pred.shape[0]*100)
    #print(roc_auc_score(y_test, y_pred, multi_class='ovr'))


if __name__ == '__main__':
    main()
