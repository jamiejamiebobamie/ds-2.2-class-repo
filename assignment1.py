"""
DS2.2 Homework 1:

https://github.com/Make-School-Courses/DS-2.2-Deep-Learning/blob/master/Assignments/HW1.ipynb

Build a linear regression and logistic regression with Keras
1- Build a Keras Model for linear regression (check: https://keras.io/activations/).
Use Boston Housing Dataset to train and test your model

2- Build a Keras Model for logistic regression. Use diabetes.csv to train and test

Comments:

1- Build the simplest model for linear regression with Keras and compare your
model performance with from sklearn.linear_model import LinearRegression

2- Build the simplest model for logistic regression with Keras and compare your
model performance with from sklearn.linear_model import LogisticRegression

3- Add more complexity to your models in (1) and (2) and compare with previous
results

https://github.com/Make-School-Courses/DS-2.2-Deep-Learning
"""

# ------- Linear Regression scikit learn on boston housing data

# https://www.geeksforgeeks.org/ml-boston-housing-kaggle-challenge-with-linear-regression/

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
boston = load_boston()
print(boston.DESCR)

boston['Price'] = boston.target
# Input Data
x = boston.data
# Output Data
y = boston.target
# splitting data to training and testing dataset.
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2,
                                                    random_state = 0)
reg = LinearRegression().fit(xtrain, ytrain)
print(reg.coef_)
print(reg.intercept_)

# ------- Keras model linear regression on boston housing data

# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

from keras.models import Sequential
from pandas import read_csv
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# commented out for performance :D!
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, x, y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ------- logistic Regression scikit learn on diabetes dataset

# https://towardsdatascience.com/end-to-end-data-science-example-predicting-diabetes-with-logistic-regression-db9bc88b4d16
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np

diabetesDF = pd.read_csv('/Users/jamesmccrory/Documents/dev/DS-2.2-Deep-Learning-master/Notebooks/Datasets/diabetes.csv')
print(diabetesDF.head())

dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]

trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds
# np.mean(trainData, axis=0) => check that new means equal 0
# np.std(trainData, axis=0) => check that new stds equal 1

diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)
accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")


# ------- Keras model logistic regression on diabetes dataset

# https://www.kaggle.com/atulnet/pima-diabetes-keras-implementation

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

pdata = diabetesDF
pdata.head()

pdata.describe()

from sklearn.model_selection import train_test_split

features = list(pdata.columns.values)
features.remove('Outcome')
print(features)
X = pdata[features]
y = pdata['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print(X_train.shape)
print(X_test.shape)

# Ensure that fieldnames aren't included
X_train = X_train.values
y_train = y_train.values
X_test  = X_test.values
y_test  = y_test.values

NB_EPOCHS = 1000  # num of epochs to test for
BATCH_SIZE = 16

## Create our model
model = Sequential()

# 1st layer: input_dim=8, 12 nodes, RELU
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
# 2nd layer: 8 nodes, RELU
model.add(Dense(8, init='uniform', activation='relu'))
# output layer: dim=1, activation sigmoid
model.add(Dense(1, init='uniform', activation='sigmoid' ))

# Compile the model
model.compile(loss='binary_crossentropy',   # since we are predicting 0/1
             optimizer='adam',
             metrics=['accuracy'])

# checkpoint: store the best model
ckpt_model = 'pima-weights.best.hdf5'
checkpoint = ModelCheckpoint(ckpt_model,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]

print('Starting training...')
# train the model, store the results for plotting
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    nb_epoch=NB_EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks_list,
                    verbose=0)


import matplotlib.pyplot as plt

# Model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
