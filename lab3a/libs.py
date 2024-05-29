import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from numpy.lib.stride_tricks import sliding_window_view
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import numpy as np

# fetch dataset
steel_industry_energy_consumption = fetch_ucirepo(id=851)

# data (as pandas dataframes)
x = steel_industry_energy_consumption.data.features
y = steel_industry_energy_consumption.data.targets

# metadata
print(steel_industry_energy_consumption.metadata)

# variable information
print(steel_industry_energy_consumption.variables)

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
    X_Seq.append(x[i:i + seq_len])
    y_Seq.append(y[i + seq_len])

y_Seq = to_categorical(y_Seq, o_num)
X_Seq = np.array(X_Seq)

X_train, X_test, y_train, y_test = train_test_split(X_Seq, y_Seq, test_size=0.2, random_state=42)


model = Sequential()
model.add(SimpleRNN(32, input_shape=(seq_len, i_num), activation='relu'))
model.add(Dense(o_num, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
print("\n\n")

from keras.layers import LSTM

model = Sequential()
model.add(LSTM(32, input_shape=(seq_len, i_num), activation='relu'))
model.add(Dense(o_num, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
print("\n\n")

from keras.layers import GRU

model = Sequential()
model.add(GRU(32, input_shape=(seq_len, i_num), activation='relu'))
model.add(Dense(o_num, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32)
