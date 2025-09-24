# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:17:04 2025

@author: Ashlesha
"""

"""
Dummy Variables

Instructions: 

Problem Statement: 
Data is one of the most important assets. It is often common that
data is stored in distinct systems with different formats and forms.
Non-numeric form of data makes it tricky while developing 
mathematical equations for prediction models. We have the 
preprocessing techniques to make the data convert to numeric
form. Explore the various techniques to have reliable uniform 
standard data, you can go through this link:
 	

1)	Prepare the dataset by performing the preprocessing techniques, 
    to have the all the features in numeric format.

| Index | Animals | Gender | Homly | Types |
| ----- | ------- | ------ | ----- | ----- |
| 1     | Cat     | Male   | Yes   | A     |
| 2     | Dog     | Male   | Yes   | B     |
| 3     | Mouse   | Male   | Yes   | C     |
| 4     | Mouse   | Male   | Yes   | C     |
| 5     | Dog     | Female | Yes   | A     |
| 6     | Cat     | Female | Yes   | B     |
| 7     | Lion    | Female | Yes   | D     |
| 8     | Goat    | Female | Yes   | E     |
| 9     | Cat     | Female | Yes   | A     |
| 10    | Dog     | Male   | Yes   | B     |


"""


import pandas as pd

# Step 1: Create the dataset
data = {
    'Index': [1,2,3,4,5,6,7,8,9,10],
    'Animals': ['Cat', 'Dog', 'Mouse', 'Mouse', 'Dog', 'Cat', 'Lion', 'Goat', 'Cat', 'Dog'],
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Male'],
    'Homly': ['Yes']*10,
    'Types': ['A','B','C','C','A','B','D','E','A','B']
}

# Load into DataFrame
df = pd.DataFrame(data)

# Step 2: Drop the Index column (not needed for modeling)
df.drop('Index', axis=1, inplace=True)

# Step 3: Convert categorical variables into dummy/indicator variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 4: Show the final DataFrame with all numeric columns
print("Preprocessed Dataset (all numeric):")
df_encoded

'''
         Animals_Dog  Animals_Goat  Animals_Lion  ...  Types_C  Types_D     Types_E
0        False         False         False        ...    False    False    False
1         True         False         False        ...    False    False    False
2        False         False         False        ...     True    False    False
3        False         False         False        ...     True    False    False
4         True         False         False        ...    False    False    False
5        False         False         False        ...    False    False    False
6        False         False          True        ...    False     True    False
7        False          True         False        ...    False    False     True
8        False         False         False        ...    False    False    False
9         True         False         False        ...    False    False    False
'''
