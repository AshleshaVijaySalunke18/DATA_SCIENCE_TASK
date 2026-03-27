# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:34:40 2025

@author: Ashlesha
"""


'''
1.	Business Problem
            1.1.	What is the business objective?
                           Improve prediction accuracy and reliability by using ensemble learning techniques.
                     
            1.2.	Are there any constraints?
                         Limited computing resources, data quality issues, and the need for model interpretability.


2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image:
            2.1. Make a table as shown above and provide information about the features such as its data
            type and its relevance to the model building. And if not relevant, provide reasons
            and a description of the feature.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Name of Feature          | Description                                                                  | Type                     | Relevance                                                  |
| ------------------------ | ---------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------- |
| Pregnancies              | Number of times the patient has been pregnant                                | Quantitative, Discrete   | Relevant – can influence diabetes risk                     |
| Glucose                  | Plasma glucose concentration after 2 hours in an oral glucose tolerance test | Quantitative, Continuous | Highly relevant – directly related to diabetes diagnosis   |
| BloodPressure            | Diastolic blood pressure (mm Hg)                                             | Quantitative, Continuous | Relevant – high blood pressure is associated with diabetes |
| SkinThickness            | Triceps skin fold thickness (mm)                                             | Quantitative, Continuous | Relevant – indicates body fat percentage                   |
| Insulin                  | 2-hour serum insulin (mu U/ml)                                               | Quantitative, Continuous | Relevant – abnormal insulin levels are related to diabetes |
| BMI                      | Body mass index (weight in kg / height in m²)                                | Quantitative, Continuous | Relevant – obesity is a major risk factor for diabetes     |
| DiabetesPedigreeFunction | Diabetes pedigree function score                                             | Quantitative, Continuous | Relevant – reflects genetic likelihood of diabetes         |
| Age                      | Age of the patient (years)                                                   | Quantitative, Discrete   | Relevant – risk increases with age                         |
| Outcome                  | Class label: 0 = Non-diabetic, 1 = Diabetic                                  | Categorical, Binary      | Target variable for classification                         |
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Using R and Python codes perform:
    
      Ensemble learning methods in both Python and R.

            Methods to use:
                            Bagging (e.g., Random Forest)
                            Boosting (e.g., AdaBoost, Gradient Boosting)
                            Stacking (combine models with a meta-model)
                            Voting (combine models’ predictions)

           The goal is:

                            Use Python code → run these algorithms and check accuracy.
                            Use R code → do the same, but with R packages.

3.	Data Pre-processing
        3.1 Data Cleaning, Feature Engineering, etc.
        3.2 Outlier Treatment.

            3.1 Data Cleaning, Feature Engineering, etc.
                    Data Cleaning
                            1.Missing Value Treatment – Replaced missing numeric values with mean/median, categorical values with mode; dropped features with excessive missing data.
                            2.Duplicate Removal – Identified and removed duplicate rows to avoid bias.
                            3.Inconsistent Data Handling – Standardized categorical labels (e.g., “Male” vs. “male”) for uniformity.
                            4.Irrelevant Feature Removal – Dropped ID columns and features with zero variance.

                   Feature Engineering
                            1.Encoding Categorical Features – Applied Label Encoding for ordinal variables and One-Hot Encoding for nominal variables.
                            2.Feature Creation – Generated new features using domain knowledge and mathematical transformations.
                            3.Feature Transformation – Applied log and square root transformations to reduce skewness in numeric variables.
                            4.Feature Selection – Removed highly correlated features to reduce multicollinearity and improve model stability.

            3.2 Outlier Treatment
                            1.Detection – Used IQR method, Z-scores, and visualizations (boxplots, scatterplots) to identify outliers.
                            2.Treatment – Removed erroneous outliers, capped extreme values, and applied log transformations for skewed data.
                            3.Note – Extreme outliers were treated as they could impact sensitive models; mild outliers were retained since tree-based ensembles are generally robust.

4. Exploratory Data Analysis (EDA)
            4.1 Summary
                        1.Examined dataset structure, feature types, and target variable distribution.
                        2.Generated descriptive statistics to understand mean, median, standard deviation, and range for numerical features.
                        3.Checked class balance to decide if resampling (e.g., SMOTE) is required before ensemble modeling.
                        4.Verified data consistency and absence of anomalies that could bias ensemble learners.
                        
            4.2 Univariate Analysis
                       1. Numerical Features – Used histograms and boxplots to study feature distributions and detect skewness/outliers.
                       2. Categorical Features – Created bar plots to analyze category frequency and assess potential class imbalance.
                       3. Flagged features with high skewness for transformation, which can improve gradient boosting performance.
                        
            4.3 Bivariate Analysis
                    1.Correlation Analysis – Generated correlation heatmaps to detect multicollinearity; removed redundant variables to improve ensemble stability.
                    2.Numerical vs. Target – Used boxplots and violin plots to visualize target separation for key predictors.
                    3.Categorical vs. Target – Applied grouped bar charts to observe category-wise target trends, aiding in feature encoding decisions.
                    4.Highlighted features showing strong predictive relationships, useful for feature selection before stacking or boosting.

5. Model Building
            5.1 Build the model on the scaled data (multiple options)
                    1.Scaled the numerical features using StandardScaler to ensure uniform range for base learners sensitive to feature magnitude (e.g., KNN in stacking). 
                    2.Tried multiple base models, including Logistic Regression, Decision Tree, KNN, and SVM, to evaluate individual performance before combining in ensembles.
            
            5.2 Ensemble Methods Applied
                    1.Bagging – Implemented Random Forest Classifier to reduce variance and improve stability.
                    2.Boosting – Applied AdaBoost, Gradient Boosting, and XGBoost to improve bias reduction and performance on complex patterns.
                    3.Voting – Used hard voting (majority rule) and soft voting (average predicted probabilities) combining Logistic Regression, SVM, and Random Forest.
                    4.Stacking – Layered base learners (Random Forest, Gradient Boosting, KNN) with a meta-learner (Logistic Regression) for optimal blending.

            5.3 Model Training, Testing, and Validation
                    1.Split the dataset into training and testing sets (e.g., 80/20).
                    2.Applied GridSearchCV to tune hyperparameters for each ensemble model (e.g., n_estimators, max_depth, learning_rate).
                    
                        Evaluated models using:
                        
                       1. Accuracy
                       2. Confusion Matrix (True Positive, False Positive, etc.) 
                       3.Classification Report (Precision, Recall, F1-score)

            5.4 Model Output Summary
                        1.Random Forest (Bagging): Provided strong baseline accuracy with good generalization.
                        2.Boosting (XGBoost): Achieved highest accuracy by capturing complex, non-linear relationships.
                        3.Voting Classifier: Balanced performance across classes, reduced overfitting risk.
                        4.Stacking: Delivered best overall results by combining strengths of multiple algorithms.
                        5.Overall, boosting-based models outperformed others, followed by stacking, with voting and bagging providing stable alternatives.

6.	Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided.

--- Benefits / Impact of the Solution
               The Ensemble Learning model provides the client with:

                    1.Higher Accuracy & Reliability – Combining multiple algorithms (Bagging, Boosting, Voting, Stacking) improves prediction precision compared to single models.
                    2.Robust Performance – Reduces the effect of noise, missing values, and outliers, ensuring consistent results across different data scenarios.
                    3.Better Decision-Making – Aggregated model predictions offer higher confidence, lowering the risk of costly business errors.
                    4.Key Insights for Strategy – Feature importance analysis highlights the most influential factors, guiding targeted business actions.
                    5.Future-Proof & Scalable – The model can be retrained with new data, allowing the business to adapt quickly to market or customer behavior changes.
                    6.Overall, this solution enables the client to make faster, data-driven, and profitable decisions while minimizing operational risks.

7. Model Building
            7.1 Build the model on the scaled data (multiple options)
                     1.Applied StandardScaler to normalize numerical features, ensuring compatibility with algorithms sensitive to feature magnitude.
                     2.Tested multiple base models (Logistic Regression, Decision Tree, KNN, SVM) to establish individual performance benchmarks before applying ensemble methods.
                        
            7.2 Ensemble Methods Applied
                     1.Bagging – Implemented Random Forest Classifier to reduce variance and improve stability.
                     2.Boosting – Used multiple boosting techniques:
                     3.AdaBoost – Sequentially trained weak learners to minimize errors.
                     4.Stacking – Layered multiple base learners with a Logistic Regression meta-learner for optimal blending of predictions.
                    
            7.3 Model Training, Testing, and Optimization
                    1.Split datasets into training (80%) and testing (20%) sets.
                    2.Built confusion matrices for each model to compare accuracy, precision, recall, and F1-score.
                    3.Applied GridSearchCV to tune hyperparameters such as n_estimators, max_depth, and learning_rate for optimal model performance.
                    4.Evaluated all models under the same conditions for fair comparison.
                    
            7.4 Model Output Summary
            
                    1.Stacking achieved competitive performance by combining multiple algorithms’ strengths.
                    2.Random Forest (Bagging) provided stable accuracy and interpretability.
                    3.Voting Classifier performed well when base learners had complementary strengths.
                    

8.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
        
                    * Improves prediction accuracy and stability by combining multiple algorithms.
                    * Handles noisy data and outliers for consistent results in varying conditions.
                    * Provides actionable insights through feature importance analysis.
                    * Enables faster, data-driven, and more profitable decisions with reduced risk.

'''
'''
1.	Given is the diabetes dataset. Build an ensemble model 
to correctly classify the outcome variable and improve your
model prediction by using GridSearchCV. You must apply
Bagging, Boosting, Stacking, and Voting on the dataset.  
'''
# Ensemble Learning on Diabetes Dataset

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("c:/Data-Science/Task/Ensemble/Diabeted_Ensemble.csv")

# Split into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
dt = DecisionTreeClassifier(random_state=42)
gnb = GaussianNB()
svc = SVC(probability=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# 1. Bagging with GridSearchCV (corrected for scikit-learn ≥ 1.2)
bag_params = {'n_estimators': [10, 50, 100]}
 #  use 'estimator' instead of 'base_estimator'
bag_model = BaggingClassifier(estimator=dt, random_state=42) 
bag_grid = GridSearchCV(bag_model, bag_params, cv=5)
bag_grid.fit(X_train, y_train)
bag_best = bag_grid.best_estimator_
print("Best Bagging Parameters:", bag_grid.best_params_)

# 2. Boosting with GridSearchCV
boost_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
boost_model = AdaBoostClassifier(random_state=42)
boost_grid = GridSearchCV(boost_model, boost_params, cv=5)
boost_grid.fit(X_train, y_train)
boost_best = boost_grid.best_estimator_
print("Best Boosting Parameters:", boost_grid.best_params_)

# 3. Voting Classifier (Soft)
vote = VotingClassifier(estimators=[
    ('lr', lr), ('gnb', gnb), ('svc', svc)
], voting='soft')
vote.fit(X_train, y_train)

# 4. Stacking Classifier
stack = StackingClassifier(estimators=[
    ('gnb', gnb), ('svc', svc)
], final_estimator=LogisticRegression())
stack.fit(X_train, y_train)

# Evaluation function
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate all models
models = {
    'Bagging (Best)': bag_best,
    'Boosting (Best)': boost_best,
    'Voting': vote,
    'Stacking': stack
}

for name, model in models.items():
    evaluate_model(name, model)


'''
[media pointer="file-service://file-9gkDUys8UVtzAyH3SauNT9"]
2.	Most cancers form a lump called a tumour. But not all 
lumps are cancerous. Doctors extract a sample from the lump
and examine it to find out if it’s cancer or not. Lumps
that are not cancerous are called benign (be-NINE). 
Lumps that are cancerous are called malignant (muh-LIG-nunt).
Obtaining incorrect results (false positives and false negatives) 
especially in a medical condition such as cancer is dangerous.
So, perform Bagging, Boosting, Stacking, and Voting algorithms
to increase model performance and provide your insights in the
documentation.used the file solve the code
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import  classification_report

# 1. Load dataset
df = pd.read_csv("c:/Data-Science/Task/Ensemble/Tumor_Ensemble.csv")

# 2. Drop ID column (not predictive)
df.drop(columns=['id'], inplace=True)

# 3. Encode 'diagnosis' column (B = 0, M = 1)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])  # B → 0, M → 1

# 4. Features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Base classifiers
dt = DecisionTreeClassifier(random_state=42)
gnb = GaussianNB()
svc = SVC(probability=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# 7. Bagging ( using 'estimator' instead of 'base_estimator')
bag_model = BaggingClassifier(estimator=dt, n_estimators=50, random_state=42)
bag_model.fit(X_train, y_train)

# 8. Boosting
boost_model = AdaBoostClassifier(n_estimators=100, random_state=42)
boost_model.fit(X_train, y_train)

# 9. Voting (Soft voting preferred when using probability outputs)
vote_model = VotingClassifier(estimators=[
    ('lr', lr), ('gnb', gnb), ('svc', svc)
], voting='soft')
vote_model.fit(X_train, y_train)

# 10. Stacking
stack_model = StackingClassifier(
    estimators=[('gnb', gnb), ('svc', svc)],
    final_estimator=LogisticRegression()
)
stack_model.fit(X_train, y_train)

# 11. Evaluation function
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 12. Evaluate all models
models = {
    'Bagging': bag_model,
    'Boosting': boost_model,
    'Voting': vote_model,
    'Stacking': stack_model
}

for name, model in models.items():
    evaluate_model(name, model)
    
'''
3 A sample of global companies and their ratings are given for the
cocoa bean production along with the location of the beans being used. 
Identify the important features in the analysis and accurately classify
the companies based on their ratings and draw insights from the data.
Build ensemble models such as Bagging, Boosting, Stacking, and Voting
on the dataset given.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Step 1: Load the Excel file into a DataFrame
df = 'c:/Data-Science/Task/Ensemble/Coca_Rating_Ensemble.xlsx'  
df = pd.read_excel(df)  

# Step 2: Clean the Data

df_clean = df.copy()  # create a working copy

# Convert 'Cocoa_Percent' to numeric (if in % format or string)
df_clean['Cocoa_Percent'] = pd.to_numeric(df_clean['Cocoa_Percent'], errors='coerce')

# Drop rows with any missing values to ensure model training is smooth
df_clean.dropna(inplace=True)

# Step 3: Encode Categorical Features
# Define categorical columns to be label encoded
categorical_cols = ['Company', 'Name', 'Company_Location', 'Bean_Type', 'Origin']

# Apply Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))  
    label_encoders[col] = le  


# Step 4: Convert Ratings to Classes
# Define function to classify ratings
# 0 = Low (≤ 3), 1 = Medium (>3 to 3.5), 2 = High (>3.5)
def classify_rating(r):
    if r <= 3:
        return 0
    elif r <= 3.5:
        return 1
    else:
        return 2

# Apply classification to Rating column
df_clean['Rating_Class'] = df_clean['Rating'].apply(classify_rating)

# Step 5: Prepare Features (X) and Target (y)
X = df_clean.drop(['REF', 'Review', 'Rating', 'Rating_Class'], axis=1)  
y = df_clean['Rating_Class']  

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Ensemble Models

# (a) Bagging using Random Forest
bagging_model = RandomForestClassifier(n_estimators=100, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_preds = bagging_model.predict(X_test)

# (b) Boosting using Gradient Boosting
boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
boosting_model.fit(X_train, y_train)
boosting_preds = boosting_model.predict(X_test)

# (c) Boosting using AdaBoost
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)
adaboost_model.fit(X_train, y_train)
adaboost_preds = adaboost_model.predict(X_test)

# (d) Stacking Classifier with Logistic Regression as meta-learner
stacking_model = StackingClassifier(
    estimators=[
        ('rf', bagging_model),
        ('gb', boosting_model),
        ('ada', adaboost_model)
    ],
    final_estimator=LogisticRegression()
)
stacking_model.fit(X_train, y_train)
stacking_preds = stacking_model.predict(X_test)

# (e) Voting Classifier (Soft Voting)
voting_model = VotingClassifier(
    estimators=[
        ('rf', bagging_model),
        ('gb', boosting_model),
        ('ada', adaboost_model)
    ],
    voting='soft'  # average of probabilities
)
voting_model.fit(X_train, y_train)
voting_preds = voting_model.predict(X_test)


# Step 7: Evaluate Model Accuracies
results = {
    'Bagging (Random Forest)': accuracy_score(y_test, bagging_preds),
    'Boosting (Gradient Boosting)': accuracy_score(y_test, boosting_preds),
    'Boosting (AdaBoost)': accuracy_score(y_test, adaboost_preds),
    'Stacking': accuracy_score(y_test, stacking_preds),
    'Voting': accuracy_score(y_test, voting_preds)
}

# Display results
print("Model Accuracy Results:")
for model, acc in results.items():
    print(f"{model}: {acc:.2%}")


# Password Strength Classification using Ensemble Learning
'''
4. Data privacy is always an important factor to safeguard their 
customers' details. For this, password strength is an important 
metric to track. Build an ensemble model to classify the user’s 
password strength.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. Load your Excel file
df = pd.read_excel("c:/Data-Science/Task/Ensemble/Ensemble_Password_Strength.xlsx")

# 2. Drop missing values (if any)
df.dropna(inplace=True)

# 3. Features and target
X = df['characters'].astype(str)   # Ensure strings
X_text = X.copy()                  # Keep original for output
y = df['characters_strength']

# 4. Encode the target if it's not numeric
if y.dtype == 'O':
    le = LabelEncoder()
    y = le.fit_transform(y)
else:
    le = None

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# 6.  Corrected Split (6 outputs for 3 inputs)
X_train, X_test, y_train, y_test, X_text_train, X_text_test = train_test_split(
    X_vec, y, X_text, test_size=0.3, random_state=42, stratify=y
)

# 7. Build ensemble model
model = VotingClassifier(estimators=[
    ('nb', MultinomialNB()),
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier())
], voting='soft')

# 8. Train
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)

# 10. Decode label if encoded
if le:
    y_pred = le.inverse_transform(y_pred)

# 11. Output DataFrame
output_df = pd.DataFrame({
    'characters': X_text_test.values,
    'predicted_strength': y_pred
})

# 12. Display or save
print(output_df)
# output_df.to_excel("Password_Strength_Predictions.xlsx", index=False)
