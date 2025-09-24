# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:31:31 2025

@author: Ashlesha
"""
'''
1. Business Problem

1.1. Business Objective:
The HR department wants to predict employee churn based on salary hikes. The goal is to estimate the churn out rate in advance, helping management make decisions to retain employees.

1.2. Constraints:

Small dataset (10 records) — may limit model accuracy.

Only one predictor variable (Salary_hike).

Assumes a linear or transformable relationship between salary hike and churn rate.
| Feature Name     | Data Type | Description/Notes                                | Relevance to Model               |
| ---------------- | --------- | ------------------------------------------------ | -------------------------------- |
| Salary\_hike     | Numeric   | Amount of salary increase in the financial year  | Predictor (Independent Variable) |
| Churn\_out\_rate | Numeric   | Percentage of employees leaving the organization | Target (Dependent Variable)      |

Both features are numeric and essential for building a regression model.

3. Data Pre-processing
3.1 Data Cleaning

Check for missing values.

Check for duplicate rows.

3.2 Outlier Treatment

Visualize using boxplots to detect outliers.
'''

'''
Problem Statement: -
A certain organization wants an early estimate of their employee 
churn out rate. So the HR department gathered the data regarding
 the employee’s salary hike and the churn out rate in a financial
 year. The analytics team will have to perform an analysis and
 predict an estimate of employee churn based on the salary hike.
 Build a Simple Linear Regression model with churn out rate as 
 the target variable. Apply necessary transformations and record 
 the RMSE and correlation coefficient values for different models.

'''


# Step 1: Import necessary libraries
import pandas as pd                  # For handling dataframes
import numpy as np                   # For numerical computations
import matplotlib.pyplot as plt      # For plotting graphs
from sklearn.linear_model import LinearRegression  # For linear regression model
from sklearn.metrics import mean_squared_error     # For evaluating RMSE
from scipy.stats import pearsonr     # For correlation coefficient

# Step 2: Load the dataset from CSV
# Make sure the path is correct and the CSV has columns: Salary_hike, Churn_out_rate
df = pd.read_csv("c:/Data-Science/Task/Simple_linear_regression/emp_data.csv")

# Check the first few rows to verify data
print(df.head())
'''
     Salary_hike  Churn_out_rate
0         1580              92
1         1600              85
2         1610              80
3         1640              75
4         1660              72
'''
# Step 3: Explore the data
print("First few rows of data:\n", df.head())
print("\nSummary statistics:\n", df.describe())
'''
Summary statistics:
        Salary_hike  Churn_out_rate
count    10.000000       10.000000
mean   1688.600000       72.900000
std      92.096809       10.257247
min    1580.000000       60.000000
25%    1617.500000       65.750000
50%    1675.000000       71.000000
75%    1724.000000       78.750000
max    1870.000000       92.000000
'''
# Visualize the relationship between Salary_hike and Churn_out_rate
plt.scatter(df['Salary_hike'], df['Churn_out_rate'], color='blue')
plt.xlabel('Salary Hike')
plt.ylabel('Churn Out Rate')
plt.title('Salary Hike vs Churn Out Rate')
plt.show()

# Step 4: Check correlation between predictor and target
corr, _ = pearsonr(df['Salary_hike'], df['Churn_out_rate'])
print("Correlation coefficient:", corr)
#Correlation coefficient: -0.911721618690911

# Step 5: Prepare data for Linear Regression
# X = predictor variable, y = target variable
X = df[['Salary_hike']]   # Double brackets to keep X as DataFrame
y = df['Churn_out_rate']  # Series

# Step 6: Build the Simple Linear Regression model
model = LinearRegression()
model.fit(X, y)   # Train the model

# Step 7: Make predictions on training data
y_pred = model.predict(X)

# Step 8: Evaluate the model
# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE:", rmse)
#RMSE: 3.9975284623377942

# Plot regression line along with actual data points
plt.scatter(X, y, color='blue')          # Actual points
plt.plot(X, y_pred, color='red')         # Regression line
plt.xlabel('Salary Hike')
plt.ylabel('Churn Out Rate')
plt.title('Simple Linear Regression')
plt.show()

# Step 9: Try Log Transformation on predictor (optional)
# Sometimes log transformation improves linearity and reduces error
X_log = np.log(df[['Salary_hike']])  # Apply natural log to Salary_hike

# Build model on transformed data
model_log = LinearRegression()
model_log.fit(X_log, y)
y_pred_log = model_log.predict(X_log)

# Evaluate transformed model
rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))
corr_log, _ = pearsonr(y, y_pred_log)

print("RMSE after log transformation:", rmse_log)
print("Correlation coefficient after log transformation:", corr_log)
#RMSE after log transformation: 3.786003613022785
#Correlation coefficient after log transformation: 0.9212077312118856

# Plot regression line with log-transformed predictor
plt.scatter(X_log, y, color='blue')          # Actual points
plt.plot(X_log, y_pred_log, color='red')     # Regression line
plt.xlabel('Log(Salary Hike)')
plt.ylabel('Churn Out Rate')
plt.title('Linear Regression with Log Transformation')
plt.show()
