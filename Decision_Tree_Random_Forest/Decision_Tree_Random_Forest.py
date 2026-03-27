# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 15:53:00 2025

@author: Ashlesha
"""
'''

1. Business Problem
1.1 Objective:
    -Predict outcomes like customer loan defaults, movie success, or employee salaries using decision-making models.
    -Provide insights for better classification and pattern identification.
1.2 Constraints:
    -Data Constraints: Handle categorical features, missing values, and outliers.
    -Model Constraints: Decision Trees risk overfitting; Random Forest can be computationally expensive.
    -Interpretability vs Accuracy: Decision Trees are simpler but less robust than Random Forests.
2. Modeling Steps
Data Preparation: Encode categorical features, handle missing data, and split data into training/testing sets.

Modeling:
 -Decision Tree: Prone to overfitting; hyperparameters like max_depth and criterion need tuning.
 -Random Forest: More accurate; tune n_estimators, max_features, and other parameters.

3. Evaluation
  -Use accuracy, confusion matrix, and F1-score to assess performance.
  -Compare clustering and predictions before and after dimensionality reduction using PCA.
  
2.Work on each feature of the dataset to create a data dictionary as displayed in the below image:
2.1 Make a table as shown above and provide information about the features such as its data type and its relevance to the model building. And if not relevant, provide reasons and a description of the feature.
Here’s a concise Data Dictionary format to summarize the features for a Decision Tree or Random Forest model, based on common datasets like heart disease classification, movie 
classification, or salary prediction.

3.Data Pre-processing
3.1 Data Cleaning 
--Handle missing values: Impute with mean/median for numerical, mode for categorical.
--Remove irrelevant features like IDs or redundant columns.
3.2Feature Engineering
--Encode categorical variables using LabelEncoder or OneHotEncoding.
--Scale numerical features with StandardScaler or MinMaxScaler.
--Select important features using Recursive Feature Elimination (RFE) or feature importance from models.

4.	Exploratory Data Analysis (EDA):
4.1.	Summary.
--Statistical Overview: Compute mean, median, min, max, standard deviation.
--Missing Values: Identify missing data using data.isnull().sum().
--Data Types: Check feature types using data.info()

4.2.Univariate analysis.
--Numerical Features: Use histograms and box plots to visualize distribution and outliers.
--Categorical Features: Use bar plots to visualize category frequency.

4.3.Bivariate analysis.
--Numerical-Numerical Relationships: Use scatter plots and correlation heatmaps.
--Categorical-Numerical Relationships: Use box plots to analyze value distribution by category.
--Categorical-Categorical Relationships: Use cross-tabulations and stacked bar charts.


5.	Model Building
5.1	Build the model on the scaled data (try multiple options).
--Split data into train and test sets using train_test_split.
--Scale data using StandardScaler if required.
--Experiment with models such as Decision Tree and Random Forest.

5.2	Perform Decision Tree and Random Forest on the given datasets.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Decision Tree Model
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

5.3	Train and Test the data and perform cross validation techniques, compare accuracies, precision and recall and explain about them.
--Use cross-validation with cross_val_score.
--Evaluate using accuracy_score, precision, recall, and confusion_matrix.

5.4	Briefly explain the model output in the documentation. 
--Decision Tree: Simple and interpretable but prone to overfitting.
--Random Forest: More accurate and robust due to reduced overfitting.
--Performance Comparison: Highlight metrics such as:
   *Accuracy improvement from Random Forest over Decision Tree.
   *Precision/recall trade-offs based on business requirements.

6. Benefits/Impact of the Solution
--Better Decision-Making: Identifies key factors for sales, risk, or customer behavior.
--Higher Accuracy: Random Forest improves predictions over simpler models.
--Risk Management: Early detection of high-risk customers minimizes losses.
--Efficiency: Automates decision-making with clear feature insights from Decision Trees.

'''
#Problem Statements:

'''1.	A cloth manufacturing company is interested to know
 about the different attributes contributing to high sales.
 Build a decision tree & random forest model with Sales as 
 target variable (first convert it into categorical variable).

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # <-- added accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("c:/Data-Science/Task/Decision_Tree_Random_Forest/Company_Data.csv")  

# 2. Convert 'Sales' into categorical (High: > 8, Low: <= 8)
df['Sales_cat'] = df['Sales'].apply(lambda x: 'High' if x > 8 else 'Low')

# 3. Encode categorical columns
categorical_cols = ['ShelveLoc', 'Urban', 'US']
df_encoded = df.copy()

le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# 4. Features and target
X = df_encoded.drop(columns=['Sales', 'Sales_cat'])  # Drop continuous sales
y = df_encoded['Sales_cat']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 6. Decision Tree model
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 7. Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 8. Evaluation reports
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
#Decision Tree Accuracy: 0.6666666666666666
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
#Random Forest Accuracy: 0.8083333333333333
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# 9. Confusion Matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', ax=ax[0], cmap='Blues')
ax[0].set_title("Decision Tree")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[1], cmap='Greens')
ax[1].set_title("Random Forest")

plt.show()

# 10. Visualize Decision Tree (optional)
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=['Low', 'High'], filled=True)
plt.title("Decision Tree Structure")
plt.show()


'''
2. Divide the diabetes data into train and test datasets
   and build a Random Forest and Decision Tree model 
   with Outcome as the output variable.
'''

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Load dataset
df = pd.read_csv("c:/Data-Science/Task/Decision_Tree_Random_Forest/diabetes.csv")

# 3. Clean column names
df.columns = df.columns.str.strip()

# 4. Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Decision Tree model
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 7. Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 8. Evaluation reports
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
#Decision Tree Accuracy: 0.6666666666666666
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
#Random Forest Accuracy: 0.8083333333333333
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# 9. Confusion Matrices side-by-side
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Decision Tree
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Decision Tree")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

# Random Forest
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title("Random Forest")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 10. Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.title("Decision Tree Structure")
plt.show()



'''3.	Build a Decision Tree & Random Forest model on
 the fraud data. 

'''
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("c:/Data-Science/Task/Decision_Tree_Random_Forest/Fraud_check.csv")

# 2. Create 'Outcome' column (Risky if <= 30000, else Good)
df['Outcome'] = df['Taxable.Income'].apply(lambda x: 'Risky' if x <= 30000 else 'Good')

# 3. Drop original 'Taxable.Income' column
df.drop('Taxable.Income', axis=1, inplace=True)

# 4. Encode categorical columns
categorical_cols = ['Undergrad', 'Marital.Status', 'Urban']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 5. Features and target
X = df.drop('Outcome', axis=1)
y = le.fit_transform(df['Outcome']) 

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Decision Tree model
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 8. Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 9. Evaluation reports
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
#Decision Tree Accuracy: 0.8
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
#Random Forest Accuracy: 0.7916666666666666
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# 10. Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', ax=ax[0], cmap='Blues')
ax[0].set_title("Decision Tree")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[1], cmap='Greens')
ax[1].set_title("Random Forest")

plt.show()

# 11. Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['Risky', 'Good'], filled=True)
plt.title("Decision Tree Structure")
plt.show()


'''
4. In the recruitment domain, HR faces the challenge of predicting
   if the candidate is faking their salary or not.
   For example, a candidate claims to have 5 years of experience
   and earns 70,000 per month working as a regional manager. The candidate
   expects more money than his previous CTC. We need a way to verify their claims
   (is 70,000 a month working as a regional manager with an experience of 5 years
   a genuine claim or does he/she make less than that?)
   Build a Decision Tree and Random Forest model with monthly income as the target variable.
'''
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("c:/Data-Science/Task/Decision_Tree_Random_Forest/HR_DT.csv")

# 2. Rename columns for convenience
df.columns = ['Position', 'Experience', 'MonthlyIncome']

# 3. Create 'SalaryStatus' column (Bluff if < 70000, else Genuine)
df['SalaryStatus'] = df['MonthlyIncome'].apply(lambda x: 'Bluff' if x < 70000 else 'Genuine')

# 4. Encode categorical columns
le = LabelEncoder()
df['Position'] = le.fit_transform(df['Position'])

# 5. Features and target
X = df[['Position', 'Experience']]
y = le.fit_transform(df['SalaryStatus'])  # 1 = Genuine, 0 = Bluff

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Decision Tree model
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 8. Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 9. Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
#Decision Tree Accuracy: 1.0
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
#Random Forest Accuracy: 1.0
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# 10. Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', ax=ax[0], cmap='Blues')
ax[0].set_title("Decision Tree")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[1], cmap='Greens')
ax[1].set_title("Random Forest")

plt.show()

# 11. Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['Bluff', 'Genuine'], filled=True)
plt.title("Decision Tree Structure")
plt.show()

# 12. Predict for new candidate (handle unseen position manually)
candidate_position = "Region Manager"
candidate_experience = 5.0

# Map position to encoding; unseen positions get a new code
pos_map = {label: idx for idx, label in enumerate(le.classes_)}
encoded_position = pos_map.get(candidate_position, len(pos_map))

candidate_features = [[encoded_position, candidate_experience]]

dt_pred_status = dt.predict(candidate_features)[0]
rf_pred_status = rf.predict(candidate_features)[0]

print(f"\nCandidate claims ₹70,000 with 5 years as {candidate_position}")
print(f"[Decision Tree] Status: {'Genuine' if dt_pred_status else 'Bluff'}")
print(f"[Random Forest] Status: {'Genuine' if rf_pred_status else 'Bluff'}")
