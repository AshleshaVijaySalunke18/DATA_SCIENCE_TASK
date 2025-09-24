# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 10:09:58 2025

@author: Ashlesha
"""
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Data-Science\Task\Multilinear_Regression\Avacado_Price.csv")

# Step 3: Preprocess Data
df_encoded = pd.get_dummies(df, columns=['type', 'region'], drop_first=True)

# Step 4: Define Features and Target
X = df_encoded.drop('AveragePrice', axis=1)
y = df_encoded['AveragePrice']

# Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Build Multilinear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
