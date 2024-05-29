from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as mp
import seaborn
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

def create_heatmap(df):
    seaborn.heatmap(df.corr(), cmap='coolwarm', annot=True)
    mp.show()

def clear_data(df):
    df['stalk-root'] = df['stalk-root'].fillna(df['stalk-root'].mode()[0])
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop('veil-type', axis=1)
    return df

def regressor_laptop():
    data = pd.read_csv('../lab2b/data/Laptop_price.csv')
    df = pd.DataFrame(data)
    df['Brand'] = pd.factorize(df['Brand'])[0]
    x = df.drop('Price', axis=1)
    y = df['Price']
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)

    mlp_reg = MLPRegressor(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam')
    mlp_reg.fit(trainX, trainY)
    y_pred = mlp_reg.predict(testX)
    df_pred_res = pd.DataFrame({'Actual': testY, 'Predicted': y_pred, 'Diff': testY - y_pred})
    print(df_pred_res.to_string())
    pd.DataFrame(mlp_reg.loss_curve_).plot(figsize=(8, 5))
    mp.show()

def classifier_mushrooms():
    mushroom = fetch_ucirepo(id=73)
    df = mushroom.data.original
    print(df['veil-type'].unique())
    df = clear_data(df)
    df = df.apply(lambda x: pd.factorize(x)[0])
    #create_heatmap(df)
    x = df.drop('poisonous', axis=1)
    y = df['poisonous']
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)

    mlp_class = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam')
    mlp_class.fit(trainX, trainY)
    y_pred = mlp_class.predict(testX)
    df_pred_res = pd.DataFrame({'Actual': testY, 'Predicted': y_pred, 'Diff': testY - y_pred})
    print(df_pred_res.to_string())
    pd.DataFrame(mlp_class.loss_curve_).plot(figsize=(8, 5))
    mp.show()
def main():
    #regressor_laptop()
    classifier_mushrooms()

if __name__ == '__main__':
    main()
