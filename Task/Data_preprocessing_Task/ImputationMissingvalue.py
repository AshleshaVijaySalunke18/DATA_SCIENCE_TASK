# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:23:36 2025

@author: Ashlesha
"""
"""
Imputation  

Instructions: 
Please share your answers filled inline in the word document. 
Submit code files wherever applicable.
	
Problem Statement:  
Majority of the datasets have missing values, that might be 
because the data collected were not at regular intervals or 
the breakdown of instruments and so on. It is nearly impossible 
to build the proper model or in other words, get accurate results. 
The common techniques are either removing those records completely 
or substitute those missing values with the logical ones, there are
 various techniques to treat these types of problems.

1)	Prepare the dataset using various techniques to solve the problem,
explore all the techniques available and use them to see 
which gives the best result.

"""
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

# Step 1: Create DataFrame with missing values from the image data
# Simulated patient data with missing values across 7 columns
data = {
    'Temp': [97.5, 98.1, 99.0, np.nan, 98.6, 97.2, np.nan, 99.3, 97.9, 98.4],
    'HR': [70, np.nan, 72, 75, 76, 74, 71, np.nan, 73, 77],
    'O2': [95, 96, np.nan, 94, 93, 92, 91, 97, np.nan, 95],
    'BP': [120, 118, np.nan, 122, 115, np.nan, 119, 121, np.nan, 117],
    'Glucose': [85, 90, 88, np.nan, 87, 84, 89, np.nan, 86, 91],
    'WBC': [5.5, np.nan, 5.8, 6.1, np.nan, 5.3, 5.9, np.nan, 6.2, np.nan],
    'Platelets': [250, 240, np.nan, 230, 255, np.nan, 260, 245, np.nan, 235]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
df
'''
   Temp    HR    O2     BP  Glucose  WBC  Platelets
0  97.5  70.0  95.0  120.0     85.0  5.5      250.0
1  98.1   NaN  96.0  118.0     90.0  NaN      240.0
2  99.0  72.0   NaN    NaN     88.0  5.8        NaN
3   NaN  75.0  94.0  122.0      NaN  6.1      230.0
4  98.6  76.0  93.0  115.0     87.0  NaN      255.0
5  97.2  74.0  92.0    NaN     84.0  5.3        NaN
6   NaN  71.0  91.0  119.0     89.0  5.9      260.0
7  99.3   NaN  97.0  121.0      NaN  NaN      245.0
8  97.9  73.0   NaN    NaN     86.0  6.2        NaN
9  98.4  77.0  95.0  117.0     91.0  NaN      235.0
'''

# Step 2: Drop all rows that have any missing value
# Useful for quick cleanup but can result in significant data loss
df_dropna = df.dropna()
print("\nDropped Missing Rows:")
df_dropna
#   Temp    HR    O2     BP   Glucose  WBC  Platelets
#0  97.5  70.0  95.0  120.0     85.0  5.5      250.0

# Step 3: Mean Imputation
# Replace missing values with the mean of each column
mean_imp = SimpleImputer(strategy='mean')
df_mean = pd.DataFrame(mean_imp.fit_transform(df), columns=df.columns)
print("\nMean Imputation:")
df_mean
'''
Temp    HR      O2          BP       Glucose  WBC  Platelets
0  97.50  70.0  95.000  120.000000     85.0  5.5      250.0
1  98.10  73.5  96.000  118.000000     90.0  5.8      240.0
2  99.00  72.0  94.125  118.857143     88.0  5.8      245.0
3  98.25  75.0  94.000  122.000000     87.5  6.1      230.0
4  98.60  76.0  93.000  115.000000     87.0  5.8      255.0
5  97.20  74.0  92.000  118.857143     84.0  5.3      245.0
6  98.25  71.0  91.000  119.000000     89.0  5.9      260.0
7  99.30  73.5  97.000  121.000000     87.5  5.8      245.0
8  97.90  73.0  94.125  118.857143     86.0  6.2      245.0
9  98.40  77.0  95.000  117.000000     91.0  5.8      235.0

'''

# Step 4: Median Imputation
# Replace missing values with the median of each column
median_imp = SimpleImputer(strategy='median')
df_median = pd.DataFrame(median_imp.fit_transform(df), columns=df.columns)
print("\nMedian Imputation:")
df_median
'''
    Temp    HR    O2     BP  Glucose   WBC  Platelets
0  97.50  70.0  95.0  120.0     85.0  5.50      250.0
1  98.10  73.5  96.0  118.0     90.0  5.85      240.0
2  99.00  72.0  94.5  119.0     88.0  5.80      245.0
3  98.25  75.0  94.0  122.0     87.5  6.10      230.0
4  98.60  76.0  93.0  115.0     87.0  5.85      255.0
5  97.20  74.0  92.0  119.0     84.0  5.30      245.0
6  98.25  71.0  91.0  119.0     89.0  5.90      260.0
7  99.30  73.5  97.0  121.0     87.5  5.85      245.0
8  97.90  73.0  94.5  119.0     86.0  6.20      245.0
9  98.40  77.0  95.0  117.0     91.0  5.85      235.0

'''

# Step 5: Mode Imputation
# Replace missing values with the most frequent value (mode) of each column
mode_imp = SimpleImputer(strategy='most_frequent')
df_mode = pd.DataFrame(mode_imp.fit_transform(df), columns=df.columns)
print("\nMode Imputation:")
df_mode

# Step 6: KNN Imputation
# Uses k-nearest neighbors to predict missing values based on similarity
knn_imp = KNNImputer(n_neighbors=2)
df_knn = pd.DataFrame(knn_imp.fit_transform(df), columns=df.columns)
print("\nKNN Imputation:")
df_knn
'''
 Temp    HR    O2     BP  Glucose  WBC  Platelets
0  97.5  70.0  95.0  120.0     85.0  5.5      250.0
1  98.1  70.0  96.0  118.0     90.0  5.3      240.0
2  99.0  72.0  95.0  115.0     88.0  5.8      230.0
3  97.2  75.0  94.0  122.0     84.0  6.1      230.0
4  98.6  76.0  93.0  115.0     87.0  5.3      255.0
5  97.2  74.0  92.0  115.0     84.0  5.3      230.0
6  97.2  71.0  91.0  119.0     89.0  5.9      260.0
7  99.3  70.0  97.0  121.0     84.0  5.3      245.0
8  97.9  73.0  95.0  115.0     86.0  6.2      230.0
9  98.4  77.0  95.0  117.0     91.0  5.3      235.0
'''

# Step 7: Iterative Imputation (MICE - Multivariate Imputation by Chained Equations)
# Predicts missing values using other features as regressors
iter_imp = IterativeImputer(random_state=42)
df_iter = pd.DataFrame(iter_imp.fit_transform(df), columns=df.columns)
print("\nIterative Imputation (MICE):")
df_iter
'''
Temp         HR         O2          BP    Glucose       WBC   Platelets
0  97.500000  70.000000  95.000000  120.000000  85.000000  5.500000  250.000000
1  98.100000  73.453831  96.000000  118.000000  90.000000  5.815630  240.000000
2  99.000000  72.000000  94.691451  119.652423  88.000000  5.800000  246.172744
3  98.261703  75.000000  94.000000  122.000000  88.332716  6.100000  230.000000
4  98.600000  76.000000  93.000000  115.000000  87.000000  5.786122  255.000000
5  97.200000  74.000000  92.000000  119.267144  84.000000  5.300000  249.033902
6  98.239474  71.000000  91.000000  119.000000  89.000000  5.900000  260.000000
7  99.300000  69.009092  97.000000  121.000000  87.575132  5.804468  245.000000
8  97.900000  73.000000  94.003621  119.316467  86.000000  6.200000  247.200285
9  98.400000  77.000000  95.000000  117.000000  91.000000  5.826065  235.000000

'''