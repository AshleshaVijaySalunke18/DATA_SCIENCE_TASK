# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 19:12:47 2025

@author: Ashlesha
"""

"""
Zero - Variance Features
Instruction

Variance measures how far a set of data is spread out. 
A variance of zero indicates that all the data values
are identical. There are various techniques to remove 
this for transforming the data into the suitable one 
for prediction.
8tyu6
Problem statement: 
Find which columns of the given dataset with zero 
variance, explore various techniques used to remove 
the zero variance from the dataset to perform certain analysis. 

1.	Work on each feature of the dataset to create a
 data dictionary as displayed in the below image:
2.	Consider the Z_dataset.csv dataset
3.	Research and perform all possible steps for obtaining solution
4.	All the codes (executable programs) should execute without errors
5.	Code modularization should be followed
6.	Each line of code 

"""

# TC1 - Data Load

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("c:/Data-Science/Task/Data_preprocessing_Task/claimants.csv", encoding='ISO-8859-1')
df
df.shape

# TC2 - Zero Variance Detection (Manual Method)
# Identify columns where values are all same
print("\n Finding columns with zero variance manually...")
manual_zero_var_cols = [col for col in df.columns if df[col].nunique() == 1 or df[col].var() == 0]
manual_zero_var_cols#Columns with zero variance

# TC3 - Column Summary (Data Dictionary)
#  Show column name, type, variance and Drop/Keep
#Creating data dictionary
def generate_data_dictionary(data):
    summary = []
    for col in data.columns:
        dtype = data[col].dtype  # type of column
        unique_vals = data[col].nunique()
        var = data[col].var() if pd.api.types.is_numeric_dtype(data[col]) else None
        treatment = "Drop" if unique_vals == 1 else "Keep"
        summary.append([col, dtype, var, treatment])
    return pd.DataFrame(summary, columns=['Column', 'DataType', 'Variance', 'Treatment'])
data_dict = generate_data_dictionary(df)
#Data dictionary created
data_dict


# TC4 - Drop Zero Variance Columns
#  Remove such columns using VarianceThreshold

#Dropping zero variance columns using sklearn
def drop_zero_variance(data, target_col=None):
    if target_col:
        X = data.drop(columns=[target_col])  # remove target before processing
    else:
        X = data.copy()

    X_numeric = X.select_dtypes(include='number')  # select numeric only
    selector = VarianceThreshold(threshold=0.0)  # remove constant cols
    reduced_data = selector.fit_transform(X_numeric)
    retained_cols = X_numeric.columns[selector.get_support()]
    df_reduced = pd.DataFrame(reduced_data, columns=retained_cols)

    if target_col:
        df_reduced[target_col] = data[target_col].values  # add target back

    return df_reduced, list(set(X_numeric.columns) - set(retained_cols))

cleaned_df, dropped_cols = drop_zero_variance(df)
dropped_cols#Columns dropped
cleaned_df.shape#New shap

# TC5 - Pipeline Integration
#  Build pipeline (zero-variance removal + model)

#Building ML pipeline
target_column = 'ATTORNEY' if 'ATTORNEY' in df.columns else None

if target_column:
    df_model = df.dropna(subset=[target_column])  # drop missing targets
    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]
    X_numeric = X.select_dtypes(include='number')  # use only numeric cols

    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('remove_zero_variance', VarianceThreshold(threshold=0.0)),
        ('model', LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)  # train model
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Pipeline trained! Accuracy: {acc:.4f}")
else:
    print("Skipped. Target column 'ATTORNEY' not found.")


# TC6 - Constant Features (Manual Method Again)

#  Check again using only nunique() == 1


constant_cols = [col for col in df.columns if df[col].nunique() == 1]#Manually finding constant columns
constant_cols#Constant columns

# TC7 - Scikit-learn VarianceThreshold (Rechecking)

# Confirm with sklearn method


df_numeric = df.select_dtypes(include='number')#Using VarianceThreshold again to confirm
selector = VarianceThreshold(threshold=0.0)
reduced = selector.fit_transform(df_numeric)
retained_cols = df_numeric.columns[selector.get_support()]
removed_cols = list(set(df_numeric.columns) - set(retained_cols))
removed_cols#Columns removed
list(retained_cols)#Columns retained


# TC8 - Preserve Target Column

# Make sure target column is not removed

def drop_zero_variance_with_target(data, target_col):#Verifying if target column is preserved
    X = data.drop(columns=[target_col])
    X_numeric = X.select_dtypes(include='number')
    selector = VarianceThreshold(threshold=0.0)
    reduced = selector.fit_transform(X_numeric)
    retained_cols = X_numeric.columns[selector.get_support()]
    df_clean = pd.DataFrame(reduced, columns=retained_cols)
    df_clean[target_col] = data[target_col].values
    return df_clean

if target_column:
    cleaned_with_target = drop_zero_variance_with_target(df, target_column)
    print("Target column preserved:", target_column in cleaned_with_target.columns)
else:
    print("Skipped. No target column.")


# TC9 - Documentation Check
# Code is modular and has clear comments

print("\nCode is modular and commented. Functions used: generate_data_dictionary(), drop_zero_variance(), etc.")

# TC10 - Export Cleaned Dataset

#Save cleaned file to CSV
cleaned_df.to_csv("cleaned_claimants.csv", index=False)#Exporting cleaned data to CSV
print("Done! File saved as 'cleaned_claimants.csv'")
