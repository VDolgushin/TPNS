import os
import sys

from sklearn.metrics import roc_auc_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from numpy.lib.stride_tricks import sliding_window_view
from keras.utils import to_categorical


from lab3b.layers.DenseLayer import DenseLayer
from lab3b.layers.RNNLayer import RNN_Layer
from lab3b.layers.Relu import ReLU
from lab3b.layers.Softmax import Softmax
from lab3b.model.Model import Model


def main():
    model = Model([
        RNN_Layer(input_size=16, hidden_size=32, output_size=16, return_sequence=True, wScale=0.4),
        ReLU(),
        RNN_Layer(input_size=16, hidden_size=12, output_size=10, return_sequence=False, wScale=0.4),
        ReLU(),
        DenseLayer(10, 3, wScale=0.05),
        Softmax()
    ])

    steel_industry_energy_consumption = fetch_ucirepo(id=851)

    x = steel_industry_energy_consumption.data.features
    y = steel_industry_energy_consumption.data.targets

    cat = x.select_dtypes(include='object').columns
    num = x.select_dtypes(include=np.number).columns
    y = LabelEncoder().fit_transform(np.ravel(y))
    column_transformer = ColumnTransformer(
        transformers=[('categorical', OneHotEncoder(), cat), ('numerical', StandardScaler(), num)])
    x = column_transformer.fit_transform(x)

    i_num = x.shape[1]
    o_num = len(np.unique(y))

    seq_len = 10

    X_Seq = []
    y_Seq = []
    for i in range(x.shape[0] - seq_len):
        X_Seq.append(x[i:i+seq_len])
        y_Seq.append(y[i + seq_len])

    y_Seq = to_categorical(y_Seq,o_num)
    X_Seq = np.array(X_Seq)

    X_train, X_test, y_train, y_test = train_test_split(X_Seq, y_Seq, test_size=0.2)

    model.train(X_train,y_train,10,0.001)
    y_pred = model.predict(X_test)


    s = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i].argmax() == y_test[i].argmax():
            s += 1
    print(s/y_pred.shape[0]*100)

    print(roc_auc_score(y_test,y_pred))

if __name__ == '__main__':
    main()