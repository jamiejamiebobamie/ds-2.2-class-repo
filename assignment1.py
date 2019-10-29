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

# ------- Linear Regression scikit learn

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
boston = load_boston()
print(boston.DESCR)

# https://www.geeksforgeeks.org/ml-boston-housing-kaggle-challenge-with-linear-regression/
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

# ------- Keras model linear regression

from keras.models import Sequential
from pandas import read_csv
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, x, y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ------- Linear Regression scikit learn
