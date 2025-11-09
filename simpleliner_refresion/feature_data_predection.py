# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:02:12 2025

@author: mohan
"""

# Import required libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

# Load the dataset (CSV file with Salary vs Experience data)
dataset = pd.read_csv(r"C:\Users\mohan\Desktop\data science naresh it\class work\data\Salary_Data.csv")

# Separate independent variable (Years of Experience) and dependent variable (Salary)
x = dataset.iloc[:, :-1]   # All rows, all columns except last → input feature
y = dataset.iloc[:, -1]    # All rows, last column → target/output

# Split dataset into training set (80%) and test set (20%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Create Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Train the model with training data
regressor.fit(x_train, y_train)

# Predict salaries for test set
y_pred = regressor.predict(x_test)

# Compare actual vs predicted values in a DataFrame
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# Visualize the regression line and test data points
plt.scatter(x_test, y_test, color='red')                         # Actual test points (red)
plt.plot(x_train, regressor.predict(x_train), color='blue')      # Regression line (blue)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()

# Get slope (coefficient) of the regression line
m = regressor.coef_
print(m)

# Get intercept of the regression line
c = regressor.intercept_
print(c)

# Predict salary for 12 years of experience
y_12 = m * 12 + c
print(y_12)

# Predict salary for 10 years of experience
y_10 = m * 10 + c
print(y_10)   


bias  = regressor.score(x_train,y_train)
print(bias)

Variance = regressor.score(x_test, y_test)
print(Variance)

dataset.mean()# This will give the mean of entire dataset

dataset['Salary'].mean()#This will give the mean peticular coloum

dataset.median() # This will give the median of the entire dataframe

dataset['Salary'].median()

dataset['Salary'].mode()

#VARIANCE

dataset.var()

dataset['Salary'].var()

# standard deviation

dataset['Salary'].std()

from scipy.stats import variation

variation(dataset.values)

#Correlation
dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness

dataset.skew()

dataset['Salary'].skew()

#Standard error

dataset.sem()

dataset['Salary'].sem()

#Z_score

import scipy.stats as stats

dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

# sum of squer regresso(SSR)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#R2 SQUARE
r_square = 1-(SSR/SST)
r_square

from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train,regressor.predict(x_train))
test_mse= mean_squared_error(y_test,y_pred)

print(train_mse)
print(test_mse)

#pickle is the frontend 
import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
  pickle.dump(regressor, file)
  print("Model has been pickled and saved as linear_regreeion_model")
#save file
import os
print(os.getcwd())
