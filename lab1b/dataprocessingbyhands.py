import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import math
import numpy


def delete_nans(df):
    for i in range(len(df)):
        for j in range(len(df.iloc[i, :])):
            if df.iloc[i, j] is None:
                df.drop(i, axis=0, inplace=True)
    return df


def clear_data(df):
    df = delete_nans(df)
    df = df.drop_duplicates(df)
    return df


def get_cov(df, index1, index2):
    res = 0
    for i in range(df.shape[0]):
        res += (df.iloc[i, index1] - df.iloc[:, index1].mean()) * (df.iloc[i, index2] - df.iloc[:, index2].mean())
    return res


def get_st_dev(df, index):
    res = 0
    for i in range(df.shape[0]):
        res += (df.iloc[i, index] - df.iloc[:, index].mean()) ** 2
    return math.sqrt(res)


def create_heatmap(df):
    corr = numpy.zeros((df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        for j in range(i, df.shape[1]):
            Sxy = get_cov(df, i, j)
            Sx = get_st_dev(df, i)
            Sy = get_st_dev(df, j)
            corr[i][j] = Sxy / (Sx * Sy)
            corr[j][i] = Sxy / (Sx * Sy)
    plt.figure(figsize=(10, 10))
    heatmap = plt.imshow(corr, cmap='coolwarm', aspect='auto', interpolation='none')
    for i in range(df.shape[1]):
        for j in range(i, df.shape[1]):
            plt.text(i, j, f'{corr[i][j]:.5f}', fontsize=16, color='black', ha='center', va='center')
            plt.text(j, i, f'{corr[i][j]:.5f}', fontsize=16, color='black', ha='center', va='center')
    plt.xticks(range(df.shape[1]), df.columns)
    plt.yticks(range(df.shape[1]), df.columns)
    plt.show()


def calc_gain_ratio(df, feature, target):
    entropy_before = calc_entropy(df[target])
    entropy_after = 0
    split_info = 0
    unique_values = df[feature].unique()
    for i in unique_values:
        split_i = df.loc[df[feature] == i]
        w_i = float(len(split_i)) / df.shape[0]
        entropy_after += calc_entropy(split_i[target])
        split_info -= w_i * math.log2(w_i)
    ig = entropy_before * len(unique_values) - entropy_after
    return ig / split_info


def calc_entropy(feature_series: pd.Series):
    entropy = 0
    for i in feature_series.value_counts():
        frequency = float(i) / len(feature_series)
        if frequency != 0:
            entropy -= frequency * math.log2(frequency)
    return entropy


def main():
    data = pd.read_csv('data/Laptop_price.csv')
    df = pd.DataFrame(data)
    print(df)
    df.info()
    df = clear_data(df)
    df['Brand'], labels = pd.factorize(df['Brand'])
    create_heatmap(df)
    df['Price'] = pd.cut(df['Price'], [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
                         labels=[0, 1, 2, 3, 4, 5, 6, 7], include_lowest=True)
    for feature in df[['Brand', 'RAM_Size', 'Storage_Capacity']].columns:
        print(f'Gain ratio for {feature}:  ', calc_gain_ratio(df, feature, 'Price'))


if __name__ == '__main__':
    main()
