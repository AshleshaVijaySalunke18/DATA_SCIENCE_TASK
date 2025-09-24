# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 10:00:15 2025

@author: Ashlesha
"""

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create the dataset (or read from CSV)
# Assuming CSV file "50_Startups.csv" is used
df = pd.read_csv("C:\Data-Science\Task\Multilinear_Regression/Startups.csv")

# Step 2a: Check first few rows
print(df.head())
'''
R&D Spend  Administration  Marketing Spend       State     Profit
0  165349.20       136897.80        471784.10    New York  192261.83
1  162597.70       151377.59        443898.53  California  191792.06
2  153441.51       101145.55        407934.54     Florida  191050.39
3  144372.41       118671.85        383199.62    New York  182901.99
4  142107.34        91391.77        366168.42     Florida  166187.94
'''
print(df.info())
print(df.describe())
'''
 R&D Spend  Administration  Marketing Spend         Profit
count      50.000000       50.000000        50.000000      50.000000
mean    73721.615600   121344.639600    211025.097800  112012.639200
std     45902.256482    28017.802755    122290.310726   40306.180338
min         0.000000    51283.140000         0.000000   14681.400000
25%     39936.370000   103730.875000    129300.132500   90138.902500
50%     73051.080000   122699.795000    212716.240000  107978.190000
75%    101602.800000   144842.180000    299469.085000  139765.977500
max    165349.200000   182645.560000    471784.100000  192261.830000
'''
# Step 3: Handle categorical variable 'State' using OneHotEncoding
# Drop one dummy variable to avoid dummy variable trap
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])], remainder='passthrough')
X = ct.fit_transform(df[['State', 'R&D Spend', 'Administration', 'Marketing Spend']])

# Target variable
y = df['Profit'].values

# Step 4: Build Multiple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 5: Predict Profit
y_pred = model.predict(X)

# Step 6: Evaluate model
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-Squared:", r2)
#Root Mean Squared Error (RMSE): 8854.761029414494
#R-Squared: 0.9507524843355148

# Step 7: Model coefficients and intercept
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
#Model coefficients: [ 1.98788793e+02 -4.18870191e+01  8.06023114e-01 -2.70043196e-022.69798610e-02]
#Model intercept: 50125.34383166715
# Step 8: Optional - Compare Actual vs Predicted Profit
plt.scatter(y, y_pred, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # Line y=x for reference
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.show()
