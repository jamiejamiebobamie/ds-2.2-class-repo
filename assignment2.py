"""Build and train a CNN+MLP deep learning model with Keras with followings
specs for MNIST dataset:

1. Conv2D(32, kernel_size=(3, 3), activation='relu')
2. Conv2D(64, kernel_size=(3, 3), activation='relu')
3. MaxPooling2D(pool_size=(2, 2))
4. Dense(128, activation='relu')
5. Dense(num_classes, activation='softmax')

Also build another model with BatchNormalization and Dropout.
Compare these two models performance for test data"""

# import numpy as np
#
# import pandas as pd
# from pandas import read_csv
#
# from sklearn.datasets import load_boston
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.externals import joblib
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
#
# boston = load_boston()
# print(boston.DESCR)
#
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.callbacks import ModelCheckpoint
# from keras.wrappers.scikit_learn import KerasRegressor



from __future__ import print_function
import keras

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import mnist

model = Sequential()
# input
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
# - hidden layers -
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Flatten())
# output
model.add(Dense(10, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    x_train /= 255
    x_test /= 255
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print(x_train[0].shape)
print(x_train[1].shape)

model.save('raw_model.h5')

history = model.fit(x_train, y_train,
                    batch_size = 32,
                    epochs = 3,
                    verbose = 1,
                    validation_split = 0.2)

te_score = model.evaluate(x_test, y_test, verbose = 0)
print('Test Loss:', te_score[0])
print('Test accuracy:', te_score[1])
