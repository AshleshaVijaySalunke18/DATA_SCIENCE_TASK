# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:03:13 2025

@author: Ashlesha
"""

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Step 2: Load the Zoo dataset
zoo = pd.read_csv("c:/Data-Science/Task/KNN/zoo.csv")  
zoo
# Step 3: Display first few rows and dataset info
zoo.head()  
zoo.info()     

# Step 4: Drop non-numeric column 'animal name' since it's not useful for prediction
zoo.drop(columns='animal name', inplace=True)

# Step 5: Split dataset into features (X) and target (y)
X = zoo.drop('type', axis=1)  # Features
y = zoo['type']               # Target variable 

# Count how many animals have hair = 1 and 0
hair_1_count = (zoo['hair'] == 1).sum()
hair_0_count = (zoo['hair'] == 0).sum()
hair_1_count
hair_0_count

# Count how many animals have feathers = 1 and 0
feathers_1_count = (zoo['feathers'] == 1).sum()
feathers_0_count = (zoo['feathers'] == 0).sum()
feathers_1_count
feathers_0_count

# Count how many animals have eggs = 1 and 0
eggs_1_count = (zoo['eggs'] == 1).sum()
eggs_0_count = (zoo['eggs'] == 0).sum()
eggs_1_count
eggs_0_count

#--HISTROGRAM

plt.hist(zoo['hair']); plt.title('hair'); plt.show()
plt.hist(zoo['feathers']); plt.title('feathers'); plt.show()
plt.hist(zoo['eggs']); plt.title('eggs'); plt.show()
plt.hist(zoo['tail']); plt.title('tail'); plt.show()
plt.hist(zoo['domestic']); plt.title('domestic'); plt.show()
plt.hist(zoo['catsize']); plt.title('catsize'); plt.show()
plt.hist(zoo['type']); plt.title('type'); plt.show()

#--pairplot
sns.pairplot(zoo[['hair', 'feathers', 'eggs', 'tail', 'domestic', 'catsize', 'type']], hue='type')
plt.show()

#Boxplot
sns.boxplot(x=zoo['hair']); plt.title('hair'); plt.show()
sns.boxplot(x=zoo['feathers']); plt.title('feathers'); plt.show()
sns.boxplot(x=zoo['eggs']); plt.title('eggs'); plt.show()
sns.boxplot(x=zoo['tail']); plt.title('tail'); plt.show()
sns.boxplot(x=zoo['domestic']); plt.title('domestic'); plt.show()
sns.boxplot(x=zoo['catsize']); plt.title('catsize'); plt.show()
sns.boxplot(x=zoo['type']); plt.title('type'); plt.show()


# Step 6: Normalize the feature data using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Scale all features between 0 and 1

# Step 7: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Initialize and train the KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)  # Fit the model on training data

# Step 9: Make predictions on the test data
y_pred = knn.predict(X_test)

# Step 10: Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))               
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))   

# Step 11: Try different values of k to find the best accuracy
acc = []  # To store accuracy values
k_range = range(3, 30, 2)  # Test odd values of k from 3 to 29

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc.append(model.score(X_test, y_test))  # Store accuracy

# Step 12: Plot Accuracy vs. K value
plt.plot(k_range, acc, marker='o', linestyle='-')
plt.title('Accuracy vs K value')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
