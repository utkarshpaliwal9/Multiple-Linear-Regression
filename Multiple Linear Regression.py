# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:36:17 2017

@author: Utkarsh
"""

import numpy as np
import pandas as pd
dataset = pd.read_csv("50_Startups.csv")

#Matrix of features
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0, strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,:3])
X[:,:3] = imputer.transform(X[:,:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])
onehotencode = OneHotEncoder(categorical_features = [3])
X = onehotencode.fit_transform(X).toarray()

#Avoid dummy variable trap
X = X[:,1:]

#SPlit
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)

#train the model and Predicting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

#Improving the model
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)), values = X[:,:], axis = 1)
x_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog= Y, exog= x_opt).fit()
regressor_OLS.summary()
regressor = LinearRegression()
x_test = np.append(arr = np.ones((10,1)), values = x_test[:,:], axis = 1)
regressor.fit(x_test[:, [0,3,5]], y_test)
y_pred = regressor.predict(x_test[:, [0,3,5]])

#Plotting y_test vs predictions
import matplotlib.pyplot as plt
plt.scatter(x_test[:,3], y_test, color = 'red')
plt.scatter(x_test[:,3], y_pred, color = 'blue')
plt.title('Predictions(Blue) vs Actual(Red)')
plt.xlabel('R&D Spend ->')
plt.ylabel('Earning ->')
plt.show()