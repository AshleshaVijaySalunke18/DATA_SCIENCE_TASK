# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:52:49 2025

@author: Ashlesha
"""
'''
Problem Statement: -

A logistics company recorded the time taken for delivery 
and the time taken for the sorting of the items for delivery.
 Build a Simple Linear Regression model to find the relationship
 between delivery time and sorting time with delivery time as
 the target variable. Apply necessary transformations and
 record the RMSE and correlation coefficient values for 
 different models.

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
df = pd.read_csv("c:/Data-Science/Task/Simple_linear_regression/delivery_time.csv")

# Step 2a: Check column names and rename if necessary
print("Columns:", df.columns)
# If columns have spaces, rename them for convenience:
df.columns = ['Delivery_Time', 'Sorting_Time']

# Step 3: Explore the data
print("First few rows:\n", df.head())
'''
First few rows:
    Delivery_Time  Sorting_Time
0          21.00            10
1          13.50             4
2          19.75             6
3          24.00             9
4          29.00            10
'''
print("\nSummary statistics:\n", df.describe())
'''

Summary statistics:
        Delivery_Time  Sorting_Time
count      21.000000     21.000000
mean       16.790952      6.190476
std         5.074901      2.542028
min         8.000000      2.000000
25%        13.500000      4.000000
50%        17.830000      6.000000
75%        19.750000      8.000000
max        29.000000     10.000000

'''
# Step 4: Scatter plot to visualize relationship
plt.scatter(df['Sorting_Time'], df['Delivery_Time'], color='blue')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Delivery Time vs Sorting Time')
plt.show()

# Step 5: Check correlation between predictor and target
corr, _ = pearsonr(df['Sorting_Time'], df['Delivery_Time'])
print("Correlation coefficient:", corr)
#Correlation coefficient: 0.8259972607955328

# Step 6: Prepare data for regression
X = df[['Sorting_Time']]       # Predictor variable as DataFrame
y = df['Delivery_Time']        # Target variable as Series

# Step 7: Simple Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = model.score(X, y)
print("Linear Regression RMSE:", rmse)
#Linear Regression RMSE: 2.7916503270617654
print("R-Squared:", r2)
#R-Squared: 0.6822714748417231

# Plot regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
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
#Log Transformation RMSE: 2.733171476682066
#Correlation coefficient (log): 0.8339325279256246

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
#Exponential Transformation RMSE: 2.9402503230562003
#Correlation coefficient (exp): 0.8085780108289262

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
#Polynomial Regression RMSE: 2.742148203780122
#Correlation coefficient (poly): 0.8327302248940076