# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:03:37 2025

@author: Ashlesha
"""
"""Duplication Typecasting
Instructions: 

Problem statement: 
Data collected may have duplicate entries, that might be because 
the data collected were not at regular intervals or any other 
reason. To build a proper solution on such data will be a tough
ask. The common techniques are either removing duplicates 
completely or substitute those values with a logical data. 
There are various techniques to treat these types of problems.

Q1. For the given dataset perform the type casting 
(convert the datatypes, ex. float to int)
Q2. Check for the duplicate values, and handle the duplicate values (ex. drop)
Q3. Do the data analysis (EDA)?

Such as histogram, boxplot, scatterplot etc
| InvoiceNo | StockCode | Description                         | Quantity | InvoiceDate    | UnitPrice | CustomerID | Country        |
| --------- | --------- | ----------------------------------- | -------- | -------------- | --------- | ---------- | -------------- |
| 536365    | 85123A    | WHITE HANGING HEART T-LIGHT HOLDER  | 6        | 12/1/2010 8:26 | 2.55      | 17850      | United Kingdom |
| 536365    | 71053     | WHITE METAL LANTERN                 | 6        | 12/1/2010 8:26 | 3.39      | 17850      | United Kingdom |
| 536365    | 84406B    | CREAM CUPID HEARTS COAT HANGER      | 8        | 12/1/2010 8:26 | 2.75      | 17850      | United Kingdom |
| 536365    | 84029G    | KNITTED UNION FLAG HOT WATER BOTTLE | 6        | 12/1/2010 8:26 | 3.39      | 17850      | United Kingdom |
| 536365    | 84029E    | RED WOOLLY HOTTIE WHITE HEART.      | 6        | 12/1/2010 8:26 | 3.39      | 17850      | United Kingdom |
| 536365    | 22752     | SET 7 BABUSHKA NESTING BOXES        | 2        | 12/1/2010 8:26 | 7.65      | 17850      | United Kingdom |
| 536365    | 21730     | GLASS STAR FROSTED T-LIGHT HOLDER   | 6        | 12/1/2010 8:26 | 4.25      | 17850      | United Kingdom |
| 536366    | 22633     | HAND WARMER UNION JACK              | 6        | 12/1/2010 8:28 | 1.85      | 17850      | United Kingdom |
| 536366    | 22632     | HAND WARMER RED POLKA DOT           | 6        | 12/1/2010 8:28 | 1.85      | 17850      | United Kingdom |

"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset from CSV file
df = pd.read_csv("c:/Data-Science/Task/Data_preprocessing_Task/onlineRetail.csv", encoding='ISO-8859-1') 


# Q1. Type Casting (convert data types)


# Convert 'Quantity' column to integer type
df['Quantity'] = df['Quantity'].astype(int)

# Convert 'UnitPrice' column to float (if not already)
df['UnitPrice'] = df['UnitPrice'].astype(float)

# Convert 'CustomerID' to integer type with support for missing values (nullable integer)
df['CustomerID'] = df['CustomerID'].astype('Int64')

# Convert 'InvoiceNo' and 'StockCode' to string type as they are identifiers
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df['StockCode'] = df['StockCode'].astype(str)

# Q2. Check and Handle Duplicate Values

# Check for duplicated rows in the dataset
duplicates = df.duplicated()

# Print the number of duplicate rows
print(f"Total duplicates: {duplicates.sum()}")
#Total duplicates: 5268

# Remove duplicate rows from the DataFrame
df = df.drop_duplicates()

# Q3. Exploratory Data Analysis (EDA)

# Histogram: Distribution of Quantity sold
plt.hist(df['Quantity'], bins=20)
plt.title('Histogram of Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

# Boxplot: Distribution and outliers in UnitPrice
sns.boxplot(x=df['UnitPrice'])
plt.title('Boxplot of UnitPrice')
plt.show()

# Scatterplot: Relationship between Quantity and UnitPrice
plt.scatter(df['Quantity'], df['UnitPrice'])
plt.title('Quantity vs UnitPrice')
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.show()
