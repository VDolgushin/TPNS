import os
from sklearn.metrics import roc_auc_score
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras import datasets
import tensorflow as tf
import keras
from keras.src.layers import AveragePooling2D, Conv2D, Flatten, Dense


def main():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train,10)
    X_test = X_test.reshape(-1,28,28,1)
    y_test = keras.utils.to_categorical(y_test,10)
    model = keras.Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1), padding='same'))
    model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3)

if __name__ == '__main__':
    main()