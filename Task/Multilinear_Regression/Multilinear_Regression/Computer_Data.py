# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 12:29:26 2025

@author: Ashlesha
"""

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 2: Load Dataset
# -----------------------------
# Make sure the file path is correct. Use raw string r"..." to avoid backslash errors.
df = pd.read_csv(r"C:\Data-Science\Task\Multilinear_Regression\computer_Data.csv")

# -----------------------------
# Step 3: Check Data
# -----------------------------
print("First 5 rows of dataset:\n", df.head())
print("\nDataset info:\n")
df.info()
print("\nDataset statistics:\n", df.describe())

# -----------------------------
# Step 4: Preprocess Data
# -----------------------------
# Convert categorical variables into numeric (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['cd','multi','premium'], drop_first=True)

# -----------------------------
# Step 5: Define Features and Target
# -----------------------------
X = df_encoded.drop('price', axis=1)  # Features
y = df_encoded['price']               # Target variable

# -----------------------------
# Step 6: Split Data into Training and Testing
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 7: Build Multilinear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 8: Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 9: Evaluate the Model
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
#Mean Squared Error (MSE): 79100.39675535809
print("Root Mean Squared Error (RMSE):", rmse)
#Root Mean Squared Error (RMSE): 281.24792755744545
print("R^2 Score:", r2)
#R^2 Score: 0.757861521556983

# -----------------------------
# Step 10: Visualize Actual vs Predicted Prices
# -----------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Computer Prices")
plt.show()

# -----------------------------
# Step 11: Feature Importance
# -----------------------------
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis')
plt.title("Feature Importance (Regression Coefficients)")
plt.show()

