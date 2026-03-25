# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:44:17 2025

@author: Ashlesha
"""
# ---  Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# --- 1. Load Dataset ---
iris = pd.read_csv("c:/Data-Science/Task/KNN/iris.csv")  
iris.head()

# --- 2. Exploratory Data Analysis (EDA) ---
iris.info()
iris.describe()
iris.isna().sum()

#  histograms
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
plt.figure(figsize=(12, 8))

for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    plt.hist(iris[feature], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {feature}")
   
# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=iris[features], palette="Set2")

#pairplot
sns.pairplot(iris, hue="Species", diag_kind="hist", palette="Set1")
plt.suptitle("Pairplot of Iris Features Colored by Species", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(iris.drop('Species', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f")

# --- 3. Data Preprocessing ---
X = iris.drop('Species', axis=1)
y = iris['Species']

# Split data with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. KNN Model Training ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# --- 5. Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)

# --- 6. Find Optimal K ---
k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_k_pred = model.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_k_pred))

# Plot Accuracy vs K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='green')
plt.title('KNN Accuracy vs K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Print optimal K
optimal_k = k_values[np.argmax(accuracies)]
print(f"\nThe highest accuracy was {np.max(accuracies):.4f} at K = {optimal_k}")
