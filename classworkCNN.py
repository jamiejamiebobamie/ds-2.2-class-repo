# from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
# from keras.models import Model
# from keras import backend as K
# import numpy as np
#
# input_img = Input(shape=(4, 4, 1))  # adapt this if using `channels_first` image data format
#
# x = Conv2D(2, (2, 2), activation='relu')(input_img)
# y = Conv2D(3, (2, 2), activation='relu')(x)
# model = Model(input_img, y)
# # cnv_ml_1 = Model(input_img, x)
#
# data = np.array([[5, 12, 1, 8], [2, 10, 3, 6], [4, 7, 9, 1], [5, 7, 5, 6]])
# data = data.reshape(1, 4, 4, 1)
# print(model.predict(data))
# print('M :')
# print(model.predict(data).reshape(3, 2, 2))
# print(model.summary())


from __future__ import print_function
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())
