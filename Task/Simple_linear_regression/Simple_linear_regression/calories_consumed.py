# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:43:16 2025

@author: Ashlesha
"""

'''
Problem Statement: -

A certain food-based company conducted a survey with the 
help of a fitness company to find the relationship between 
a person’s weight gain and the number of calories they 
consumed in order to come up with diet plans for these
 individuals. Build a Simple Linear Regression model with 
 calories consumed as the target variable. Apply necessary
 transformations and record the RMSE and correlation coefficient
 values for different models. 
'''
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures

# Step 2: Load the dataset from CSV
df = pd.read_csv("c:/Data-Science/Task/Simple_linear_regression/calories_consumed.csv")

# Step 2a: Rename columns for easier access
df.columns = ['Weight_gained', 'Calories_Consumed']

# Step 3: Explore the data
print("First few rows of the dataset:\n", df.head())
'''
First few rows of the dataset:
    Weight_gained  Calories_Consumed
0            108               1500
1            200               2300
2            900               3400
3            200               2200
4            300               2500
'''
print("\nSummary statistics:\n", df.describe())
'''

Summary statistics:
        Weight_gained  Calories_Consumed
count      14.000000          14.000000
mean      357.714286        2340.714286
std       333.692495         752.109488
min        62.000000        1400.000000
25%       114.500000        1727.500000
50%       200.000000        2250.000000
75%       537.500000        2775.000000
max      1100.000000        3900.000000

'''

# Step 4: Scatter plot to visualize relationship
plt.scatter(df['Weight_gained'], df['Calories_Consumed'], color='blue')
plt.xlabel('Weight Gained (grams)')
plt.ylabel('Calories Consumed')
plt.title('Weight Gained vs Calories Consumed')
plt.show()

# Step 5: Check correlation between predictor and target
corr, _ = pearsonr(df['Weight_gained'], df['Calories_Consumed'])
print("Correlation coefficient:", corr)
#Correlation coefficient: 0.946991008855446

# Step 6: Prepare data for regression
X = df[['Weight_gained']]       # Predictor variable as DataFrame
y = df['Calories_Consumed']     # Target variable as Series

# Step 7: Simple Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = model.score(X, y)
print("Linear Regression RMSE:", rmse)
#Linear Regression RMSE: 232.8335007096089
print("R-Squared:", r2)
#R-Squared: 0.8967919708530552

# Plot regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Weight Gained')
plt.ylabel('Calories Consumed')
plt.title('Simple Linear Regression')
plt.show()

# Step 8: Log Transformation (log of predictor)
X_log = np.log(X)
model_log = LinearRegression()
model_log.fit(X_log, y)
y_pred_log = model_log.predict(X_log)
rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))
corr_log, _ = pearsonr(y, y_pred_log)
print("Log Transformation RMSE:", rmse_log)
print("Correlation coefficient (log):", corr_log)
#Log Transformation RMSE: 253.55804039366254
#Correlation coefficient (log): 0.9368036903364728

# Step 9: Exponential Transformation (log of target)
y_log = np.log(y)
model_exp = LinearRegression()
model_exp.fit(X, y_log)
y_pred_exp_log = model_exp.predict(X)
y_pred_exp = np.exp(y_pred_exp_log)
rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
corr_exp, _ = pearsonr(y, y_pred_exp)
print("Exponential Transformation RMSE:", rmse_exp)
print("Correlation coefficient (exp):", corr_exp)
#Exponential Transformation RMSE: 272.4207117048489
#Correlation coefficient (exp): 0.9306443934300498

# Step 10: Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)
rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
corr_poly, _ = pearsonr(y, y_pred_poly)
print("Polynomial Regression RMSE:", rmse_poly)
print("Correlation coefficient (poly):", corr_poly)
#Polynomial Regression RMSE: 220.03995629653116
#Correlation coefficient (poly): 0.9527971171047488