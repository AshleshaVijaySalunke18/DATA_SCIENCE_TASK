# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:22:55 2025

@author: Ashlesha
"""
'''
1. Business Problem

     1.1. Business Objective:
                             --Build a classification model using Naïve Bayes.
                             --Predict the target category based on input features.
                             --Achieve high accuracy and reduce misclassification.
                             --Support decision-making through reliable predictions.

     1.2. Constraints:
                            --Dataset may be limited or imbalanced.
                            --Model should be fast and resource-efficient.
                            --Limited computational resources for deployment.
                            
2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image:

2.1 Make a table as shown above and provide information about the features such as its data type and its relevance to the model building. And if not relevant, provide reasons and a description of the feature.                    
 ------------------------------------------------------------------------------------------------------------------------------------                           
| Name of Feature |             Description       |          Type            |          Relevance                                   |
| --------------- | ----------------------------- | ------------------------ | ---------------------------------------------------- |
| ID              | Customer ID                   | Quantitative, Nominal    | Irrelevant, ID does not provide useful information   |
| Age             | Age of the customer           | Quantitative, Continuous | Relevant, helps identify patterns in target variable |
| Gender          | Male/Female indicator         | Qualitative, Nominal     | Relevant, can affect prediction outcomes             |
| Income          | Annual income of the customer | Quantitative, Continuous | Relevant, may correlate with target variable         |
| Purchased       | Target variable: 0/1 outcome  | Qualitative, Nominal     | Relevant, dependent variable for model               |
 ------------------------------------------------------------------------------------------------------------------------------------                           

3.	Data Pre-processing
3.1 Data Cleaning, Feature Engineering, etc.

       Data Cleaning
          --Handle Missing Values: Identify and fill or remove missing data using techniques like mean/median imputation or deletion.
          --Remove Duplicates: Detect and delete duplicate rows to avoid bias and redundancy.
          --Correct Inconsistencies: Fix errors in data such as inconsistent formats, typos, and outliers.
      
      Feature Engineering
            --Create New Features: Derive new variables from existing data, like extracting date parts or combining features.
            --Transform Features: Apply scaling, normalization, or log transformation to make data more suitable for modeling.
           --Encode Categorical Variables: Convert categorical data into numeric form using methods like one-hot encoding or label encoding.

4. Exploratory Data Analysis (EDA)

        4.1 Summary:
           --Provide an overall summary of the dataset including shape, data types, missing values, and basic statistics like mean, median, and standard deviation.
        
        4.2 Univariate Analysis:
           --Analyze individual features one at a time using visualizations like histograms, boxplots, and bar charts to understand their distribution and detect outliers.
        
        4.3 Bivariate Analysis:
          --Examine relationships between two variables using scatter plots, correlation matrices, and cross-tabulations to identify trends and associations.


5. Model Building
        5.1 Build the model on the scaled data (try multiple options):
        Apply data scaling techniques (like StandardScaler or MinMaxScaler) and train different machine learning models such as Logistic Regression, SVM, Decision Trees, or Random Forests on the scaled dataset to find the best performer.
        
        5.2 Build a Naïve Bayes model:
        Train a Naïve Bayes classifier (e.g., GaussianNB, MultinomialNB) on the training data to leverage probabilistic modeling, especially useful for text or categorical data.
        
        5.3 Validate the model with test data and obtain evaluation metrics:
        Use the test set to evaluate the model by generating a confusion matrix, and calculate precision, recall, and accuracy to measure performance.
        
        5.4 Tune the model and improve accuracy:
        Optimize hyperparameters through techniques like Grid Search or Random Search, apply feature selection, or try ensemble methods to enhance the model’s accuracy and generalization.
        

6. Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?

        1. Enables data-driven decisions to improve business efficiency and reduce risks.
        2. Optimizes resource allocation and enhances customer satisfaction with accurate predictions.
        3. Saves time and cost by automating analysis and provides actionable insights for growth.

'''



#1 Salary Dataset – Gaussian Naive Bayes
'''
Problem Statement:
1.) Prepare a classification model using the Naive
    Bayes algorithm for the salary dataset. Train and
    test datasets are given separately. Use both 
    for model building. 
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load train and test datasets
test = pd.read_csv("c:/Data-Science/Task/Naive_Bayes_Classification/SalaryData_Test.csv")
train = pd.read_csv("c:/Data-Science/Task/Naive_Bayes_Classification/SalaryData_Train.csv")

# Combine datasets for uniform encoding
combined = pd.concat([train, test], axis=0)

# Encode categorical variables
le = LabelEncoder()
for col in combined.select_dtypes(include='object').columns:
    combined[col] = le.fit_transform(combined[col].astype(str))

# Split back
train_encoded = combined.iloc[:train.shape[0], :]
test_encoded = combined.iloc[train.shape[0]:, :]

# Features and target
X_train = train_encoded.drop('Salary', axis=1)
y_train = train_encoded['Salary']
X_test = test_encoded.drop('Salary', axis=1)
y_test = test_encoded['Salary']

# Model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


#2. Social Network Ads – Bernoulli Naive Bayes
'''
Problem Statement: -
This dataset contains information of users in a social network.
This social network has several business clients which can 
post ads on it. One of the clients has a car company which 
has just launched a luxury SUV for a ridiculous price. 
Build a Bernoulli Naïve Bayes model using this dataset 
and classify which of the users of the social network
are going to purchase this luxury SUV. 1 implies that 
there was a purchase and 0 implies there wasn’t a purchase.

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
df = pd.read_csv("c:/Data-Science/Task/Naive_Bayes_Classification/NB_Car_Ad.csv")

# Drop User ID (not relevant)
df.drop('User ID', axis=1, inplace=True)

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


#3. Disaster Tweet Classification – Multinomial Naive Bayes

'''
Problem Statement: -
In this case study, you have been given Twitter data collected from an anonymous twitter handle. With the help of a Naïve Bayes model, predict if a given tweet about a real disaster is real or fake.
1 = real tweet and 0 = fake tweet

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
df = pd.read_csv("c:/Data-Science/Task/Naive_Bayes_Classification/Disaster_tweets_NB.csv")

# Check the column names
print(df.columns)

# Define features and target
X = df['text']            # Assuming the tweet text is in the 'text' column
y = df['target']          # Assuming the label is in the 'target' column (1 = real, 0 = fake)

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Predict
y_pred = mnb.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
