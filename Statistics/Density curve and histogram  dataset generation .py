# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:19:22 2025

@author: Ashlesha
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Step 1: Generate data
# =========================
np.random.seed(42)  # reproducibility
heights = np.random.normal(loc=160, scale=10, size=1000)  # mean=160, std=10, n=1000

# Create DataFrame
df = pd.DataFrame({'Height_cm': heights})

# Save to CSV
df.to_csv('height_data_normal.csv', index=False)
print("CSV file 'height_data_normal.csv' created successfully.\n")

# =========================
# Step 2: Calculate statistics
# =========================
mean_height = df['Height_cm'].mean()
mean_height
#160.19332055822323

median_height = df['Height_cm'].median()
median_height
#160.2530061223489

std_dev_height = df['Height_cm'].std()
std_dev_height
#9.792159381796756

min_height = df['Height_cm'].min()
min_height
#127.58732659930928

max_height = df['Height_cm'].max()
max_height
#198.52731490654722

# =========================
# Step 3: Plot histogram + density curve
# =========================
plt.figure(figsize=(8,5))
sns.histplot(df['Height_cm'], bins=30, kde=True, color='skyblue', edgecolor='black')

plt.title('Height Distribution with Density Curve', fontsize=14)
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
