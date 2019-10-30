from keras.layers import Lambda, Input
from keras.models import Model

import numpy as np

input = Input(shape=(3,))
double = Lambda(lambda x: 2 * x)(in)

model = Model(input=inp, output=double)
# model.compile(optimizer='sgd', loss='mse')

data = np.array([[5, 12, 1]])
print(model.predict(data))
