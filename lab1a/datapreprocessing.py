import pandas as pd
import matplotlib.pyplot as mp
import seaborn
from sklearn import tree
def clear_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def create_heatmap(df):
    seaborn.heatmap(df.corr(),cmap='coolwarm',annot=True)
    mp.show()

def main() :
    data = pd.read_csv('../lab1b/data/Laptop_price.csv')
    df = pd.DataFrame(data)
    print(df)
    df.info()
    df = clear_data(df)
#['Asus','Acer','Lenovo','HP','Dell'],
    df['Brand'], labels = pd.factorize(df['Brand'])
    print(df)
    create_heatmap(pd.DataFrame(df))
    print("-----------------------")

    clf = tree.DecisionTreeClassifier(criterion='gini')
    df['Price'] = pd.cut(df['Price'],[0,5000,10000,15000,20000,25000,30000,35000,40000],labels=[0,1,2,3,4,5,6,7],include_lowest=True)
    clf.fit(df[['Brand','RAM_Size','Storage_Capacity']],df['Price'])

    importances = clf.feature_importances_
    gain_ratios = {}
    for i, feature in enumerate(df[['Brand','RAM_Size','Storage_Capacity']].columns):
        gain = importances[i]
        split = clf.tree_.impurity[i]
        gain_ratio = gain / (split + 1e-7)
        print(f'Gain ratio for {feature}: {gain_ratio:.3f}')
        gain_ratios[feature] = gain_ratio

if __name__ == '__main__':
    main()