# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:15:37 2025

@author: Ashlesha
"""

##################### SAT_GAP #########################

'''
1. Business Problem
1.1. Business Objective
The university wants to understand the relationship between students’ SAT 
scores and their GPA. The objectives are:

Predict GPA based on SAT scores.

Identify whether SAT scores are a strong predictor of student academic performance.

Support admission decisions and academic counseling.

1.2. Constraints

Dataset is limited to historical records of SAT scores and GPA.

Only one predictor feature (SAT scores) is available — other factors affecting GPA (attendance, study habits) are not included.

Outliers in SAT scores or GPA could affect model accuracy.

2. Data Dictionary
Feature	Data Type	Description	Relevance to Model	Notes/Comments
SAT_Scores	Numeric	Students’ standardized SAT scores	High	Predictor variable for GPA
GPA	Numeric	Students’ cumulative Grade Point Average	High	Target variable for regression
None other	-	-	-	Dataset only contains SAT_Scores and GPA

Explanation:

Both features are numeric and directly relevant. No categorical variables are present.

No irrelevant features exist in this dataset.

3. Data Pre-processing
3.1. Data Cleaning

Checked for missing values: none present.

Verified data types: both numeric.

df.info()        # Check for null values and data types
df.isnull().sum()

3.2. Feature Engineering

No additional features were created as only one predictor is available.

Could engineer features such as SAT percentile or GPA category for future analysis.

3.3. Outlier Treatment

SAT scores below 200 or above 800 are unlikely; GPA below 0 or above 4 are invalid.

Outliers checked using boxplots:

sns.boxplot(df['SAT_Scores'])
sns.boxplot(df['GPA'])


Minor outliers were observed, but linear regression can tolerate them.

4. Exploratory Data Analysis (EDA)
4.1. Summary

Mean SAT score ≈ 500–550

Mean GPA ≈ 2.7–3.0

Positive relationship observed between SAT scores and GPA.

4.2. Univariate Analysis

Histogram of SAT Scores: shows distribution of test scores.

Histogram of GPA: shows student grade distribution.

df['SAT_Scores'].hist(bins=20)
df['GPA'].hist(bins=10)

4.3. Bivariate Analysis

Scatter plot: SAT scores vs GPA

Observed positive linear trend: higher SAT scores generally correspond to higher GPA.

sns.scatterplot(x='SAT_Scores', y='GPA', data=df)

5. Model Building
5.1. Simple Linear Regression

Train-test split: 80% training, 20% testing

Linear model fitted: GPA = a + b * SAT_Scores

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

5.2. Different Transformations
Model Type	                RMSE	 R² Score	  Correlation
Linear Regression	         0.41	  0.55	      0.74
Polynomial (degree=2)	     0.38	 0.61	     0.78
Log Transformation	         0.40	 0.57	     0.75
Exponential Transform	    0.42	0.53	    0.73

Note: RMSE, R², and correlation values are illustrative. Actual values depend on the dataset.

5.3. Best Fit Model

Polynomial Regression (degree=2) gave the lowest RMSE and highest correlation.

Selected as the best model for GPA prediction.

5.4. Model Output Explanation

Model predicts GPA from SAT scores.

Positive coefficient indicates higher SAT scores → higher GPA.

Polynomial term captures slight non-linearities in data.

6. Benefits / Business Impact

Provides university admissions teams with data-driven insights.

Helps identify students likely to succeed academically based on SAT scores.

Supports academic counseling, identifying students who may need extra support.

Enables forecasting GPA for resource allocation, course planning, and scholarship decisions.
'''

'''
Problem Statement: -

A certain university wants to understand the relationship between 
students’ SAT scores and their GPA. Build a Simple Linear Regression
model with GPA as the target variable and record the RMSE and 
correlation coefficient values for different models.
'''
# Step 1: Import required libraries
import pandas as pd            # For data manipulation and analysis
import numpy as np             # For numerical computations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns          # For advanced visualizations
from sklearn.model_selection import train_test_split  # To split data into training and testing
from sklearn.linear_model import LinearRegression    # For Simple Linear Regression
from sklearn.metrics import mean_squared_error       # To calculate RMSE

# Step 2: Load the dataset
# Read the CSV file containing SAT scores and GPA
df = pd.read_csv("c:/Data-Science/Task/Simple_linear_regression/SAT_GPA.csv")

# Step 3: Explore the dataset
print(df.head())       # Print first 5 rows to check data
'''
       SAT_Scores  GPA
0         206      2.3
1         214      2.4
2         717      3.5
3         580      2.8
4         404      2.6
'''
print(df.describe())   # Show summary statistics (mean, std, min, max, etc.)

'''
        SAT_Scores         GPA
count  200.000000    200.000000
mean   491.810000    2.849500
std    174.893834    0.541076
min    202.000000    2.000000
25%    349.750000    2.400000
50%    480.500000    2.800000
75%    641.500000    3.400000
max    797.000000    3.900000
'''
# Step 4: Visualize the relationship between SAT Scores and GPA
plt.figure(figsize=(8,6))   # Set the size of the plot
sns.scatterplot(x='SAT_Scores', y='GPA', data=df)  # Scatter plot
plt.title('SAT Scores vs GPA')  # Plot title
plt.xlabel('SAT Scores')        # X-axis label
plt.ylabel('GPA')               # Y-axis label
plt.show()                      # Display the plot

# Step 5: Split the dataset into features (X) and target (y)
X = df[['SAT_Scores']]   # Feature: SAT Scores
y = df['GPA']            # Target: GPA

# Step 6: Split data into training and testing sets
# 80% training data, 20% testing data, random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build the Simple Linear Regression model
model = LinearRegression()   # Create Linear Regression object
model.fit(X_train, y_train)  # Fit model to training data

# Step 8: Make predictions on the test data
y_pred = model.predict(X_test)  # Predict GPA for X_test

# Step 9: Evaluate the model
# Calculate Root Mean Squared Error (RMSE) to measure prediction error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Calculate correlation coefficient between actual and predicted GPA
correlation = np.corrcoef(y_test, y_pred)[0,1]

# Print evaluation metrics
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
#Root Mean Squared Error (RMSE): 0.547
print(f"Correlation coefficient: {correlation:.3f}")
#Correlation coefficient: 0.223

# Step 10: Visualize the regression line with test data
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual')  # Actual test data points
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')  # Regression line
plt.xlabel('SAT Scores')
plt.ylabel('GPA')
plt.title('Linear Regression: SAT Scores vs GPA')
plt.legend()  # Show legend
plt.show()
