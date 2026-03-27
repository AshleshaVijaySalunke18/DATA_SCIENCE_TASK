# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 12:51:16 2025

@author: Ashlesha
"""
# -----Hypotheses----------------
            
'''Formulate statistical hypotheses (Null & Alternative):
 1. Null Hypothesis (H₀): There is no significant difference in the
proportion of male vs. female customers across different days of the week.

2. Alternative Hypothesis (H₁): There is a significant difference
in the proportion of male vs. female customers across different days of the week.
'''

'''
1. Business Problem
Businesses need to test assumptions about populations to check if 
observed differences or patterns are statistically significant or due to chance.
1.1. Objective
To apply hypothesis testing for validating assumptions,
comparing groups, and supporting decision-making with data evidence.
1.2. Constraints (if any)
Limited data, statistical assumptions (normality, independence, 
equal variance), and fixed significance level (usually 5%).


2. Data Pre-processing
2.1 Data Cleaning, Feature Engineering, EDA etc.

Data Cleaning: Handle missing values, remove duplicates, and correct inconsistencies.
Feature Engineering: Transform variables, create new features, and encode categorical data.
EDA (Exploratory Data Analysis): Summarize data, visualize distributions, and identify patterns or outliers.

3. Model Building
Models are built to test hypotheses or make predictions by applying
appropriate statistical or ML algorithms.
3.1 Partition the dataset
Split data into training and testing sets to avoid overfitting 
and ensure unbiased evaluation.
3.2 Model(s) – Reasons to choose any algorithm
Choose algorithms (t-test, ANOVA, Chi-square, regression, 
classifiers) based on problem type, data type, and business objective.
3.3 Model(s) Improvement steps
Apply feature scaling, hyperparameter tuning, cross-validation,
or ensemble methods to improve performance.
3.4 Model Evaluation
Evaluate models using accuracy, p-values, confidence intervals,
confusion matrix, or error metrics.
3.5 Python and R codes
Implement hypothesis testing and models using Python 
(scipy, statsmodels, sklearn) and R (t.test(), aov(), chisq.test()).

4. Deployment
Deployment ensures the tested model or hypothesis results are accessible
for decision-making in real-world use.
4.1 Deploy solutions using R Shiny and Python Flask
R Shiny: Build interactive dashboards for visualization and statistical test results.
Python Flask: Create lightweight web apps or APIs to serve hypothesis
 testing outcomes and models.
'''
'''
1.)	A F&B manager wants to determine whether there is any significant 
difference in the diameter of the cutlet between two units.
A randomly selected sample of cutlets was collected from
both units and measured? Analyze the data and draw inferences
at 5% significance level. Please state the assumptions and 
tests that you carried out to check validity of the assumptions.

File: Cutlets.csv

'''

# 1. Import libraries
import pandas as pd
from scipy import stats

# 2. Load dataset
# Suppose the dataset is in "Cutlets.csv" with columns 'Unit A' and 'Unit B'
df = pd.read_csv("C:/Data-Science/Task/Hypothesis_Testing/Cutlets.csv")

# 3. Extract data
unitA = df['Unit A'].dropna()
unitB = df['Unit B'].dropna()

# 4. Check assumptions
# 4.1 Normality test (Shapiro-Wilk)
shapiro_A = stats.shapiro(unitA)
shapiro_B = stats.shapiro(unitB)

print("Shapiro-Wilk Test (Unit A):", shapiro_A)
# ShapiroResult(statistic=0.9649459719657898
# pvalue=0.31998491287231445)

print("Shapiro-Wilk Test (Unit B):", shapiro_B)
# ShapiroResult(statistic=0.9727305769920349
# pvalue=0.5225146412849426)

# 4.2 Test for equal variances (Levene’s Test)
levene_test = stats.levene(unitA, unitB)
print("Levene’s Test for Equal Variances:", levene_test)
# Levene’s Test for Equal Variances: LeveneResult(statistic=0.6650897638632386,  pvalue=0.417616221250256)

# 5. Two-sample t-test

# If Levene p > 0.05 → equal_var=True, else use Welch’s t-test
t_stat, p_value = stats.ttest_ind(unitA, unitB, equal_var=True)
print("T-test result: t-statistic = {:.4f}, p-value = {:.4f}".format(t_stat, p_value))
# T-test result: t-statistic = 0.7229, p-value = 0.4722

# 6. Alpha

alpha = 0.05
if p_value < alpha:
    print("Reject H0 → Significant difference in cutlet diameters between Unit A and Unit B.")
else:
    print("Fail to reject H0 → No significant difference in cutlet diameters between Unit A and Unit B.")

'''
Fail to reject H0 → No significant difference in cutlet diameters between Unit A
 and Unit B.
'''


'''
2.)	A hospital wants to determine whether there is any difference in the 
    average Turn Around Time (TAT) of reports of the laboratories on
    their preferred list. They collected a random sample and recorded 
    TAT for reports of 4 laboratories. TAT is defined as sample collected
    to report dispatch.Analyze the data and determine whether there is 
    any difference in average TAT among the different laboratories at
    5% significance level.
File: LabTAT.csv

'''
import pandas as pd
import scipy.stats as stats

# Load dataset
df = pd.read_csv("C:/Data-Science/Task/Hypothesis_Testing/lab_tat_updated.csv")

# Extract lab columns (adjust names as per print(df.columns) output)
lab1 = df['Laboratory_1']
lab2 = df['Laboratory_2']
lab3 = df['Laboratory_3']
lab4 = df['Laboratory_4']

# One-Way ANOVA
f_stat, p_val = stats.f_oneway(lab1, lab2, lab3, lab4)

print("F-Statistic:", f_stat)
#F-Statistic: 121.39264646442368

print("P-Value:", p_val)
#P-Value: 2.143740909435053e-58

# Hypothesis conclusion
if p_val < 0.05:
    print("Reject H0 → Significant difference in average TAT among labs.")
else:
    print("Fail to Reject H0 → No significant difference in average TAT among labs.")
#Reject H0 → Significant difference in average TAT among labs.


'''
3.)	Sales of products in four different regions is tabulated for males 
    and females. Find if male-female buyer rations are similar across regions.
East West North South

Males	50	142	131	70
Females	550	351	480	350

'''
import pandas as pd
import scipy.stats as stats

# Contingency table
data = [[50, 142, 131, 70],
        [550, 351, 480, 350]]

# Perform Chi-square test
chi2, p_val, dof, expected = stats.chi2_contingency(data)

print("Chi-square Statistic:", chi2)
#Chi-square Statistic: 80.27295426602495

print("Degrees of Freedom:", dof)
#Degrees of Freedom: 3

print("P-Value:", p_val)
#P-Value: 2.682172557281901e-17

print("\nExpected Frequencies:\n", expected)
'''
Expected Frequencies:
 [[111.01694915  91.21892655 113.05225989  77.71186441]
 [488.98305085 401.78107345 497.94774011 342.28813559]]
'''
# p_val
if p_val < 0.05:
    print("Reject H0 → Male-Female ratios differ across regions.")
else:
    print("Fail to Reject H0 → Male-Female ratios are similar across regions.")
#Reject H0 → Male-Female ratios differ across regions.


'''
4.)	Telecall uses 4 centers around the globe to process customer
    order forms. They audit a certain % of the customer order forms.
    Any error in order form renders it defective and must be reworked 
    before processing. The manager wants to check whether 
    defective % varies by center. Please analyze the data at 
    5% significance level and help the manager draw appropriate 
    inferences
File: Customer OrderForm.csv

'''
import pandas as pd
import scipy.stats as stats

# Load dataset
df = pd.read_csv("C:/Data-Science/Task/Hypothesis_Testing/CustomerOrderform.csv")

# Check structure
print(df.head())
'''
   Phillippines   Indonesia       Malta       India
0   Error Free  Error Free   Defective  Error Free
1   Error Free  Error Free  Error Free   Defective
2   Error Free   Defective   Defective  Error Free
3   Error Free  Error Free  Error Free  Error Free
4   Error Free  Error Free   Defective  Error Free
'''
print(df.columns)
#Index(['Phillippines', 'Indonesia', 'Malta', 'India'], dtype='object')

# Build contingency table: count of Error Free / Defective per center
table = df.apply(pd.Series.value_counts).T   # Transpose so centers are rows

print("\nContingency Table:\n", table)
'''
Contingency Table:
                 Error Free  Defective
Phillippines         271         29
Indonesia            267         33
Malta                269         31
India                280         20
'''
# Perform Chi-square test
chi2, p_val, dof, expected = stats.chi2_contingency(table)

print("\nChi-square Statistic:", chi2)
#Chi-square Statistic: 3.8589606858203545
print("Degrees of Freedom:", dof)
#Degrees of Freedom: 3
print("P-Value:", p_val)
#P-Value: 0.27710209912331435
print("\nExpected Frequencies:\n", expected)
'''
Expected Frequencies:
 [[271.75  28.25]
 [271.75  28.25]
 [271.75  28.25]
 [271.75  28.25]]
'''
# Interpretation
if p_val < 0.05:
    print("Reject H0 → Defective % differs significantly among centers.")
else:
    print("Fail to Reject H0 → Defective % is similar across centers.")
    
#Fail to Reject H0 → Defective % is similar across centers.

'''
5.)	Fantaloons Sales managers commented that % of males versus
    females walking into the store differ based on day of the 
    week. Analyze the data and determine whether there is evidence
    at 5 % significance level to support this hypothesis.
File: Fantaloons.csv

'''
import pandas as pd
import scipy.stats as stats

# Load Fantaloons data
df = pd.read_csv("C:/Data-Science/Task/Hypothesis_Testing/Fantaloons.csv")

# Inspect dataset
print(df.head())
'''
    Weekdays Weekend
0     Male  Female
1   Female    Male
2   Female    Male
3     Male  Female
4   Female  Female
'''
# Contingency table: Weekdays vs Weekend
contingency = pd.crosstab(df['Weekdays'], df['Weekend'])
print("\nContingency Table:\n", contingency)
'''
weekdays   female  male
female      167   120
male        66     47
'''
# Chi-Square test of independence
chi2, p, dof, expected = stats.chi2_contingency(contingency)

print("\nChi-Square Statistic:", chi2)
#Chi-Square Statistic: 0.0
print("Degrees of Freedom:", dof)
#Degrees of Freedom: 1
print("P-value:", p)
#P-value: 1.0
print("\nExpected Frequencies:\n", expected)
'''
Expected Frequencies:
 [[167.1775 119.8225]
 [ 65.8225  47.1775]]
'''
# Decision at 5% significance level
alpha = 0.05
if p < alpha:
    print("\n Reject Null Hypothesis: Gender distribution differs by day of the week.")
else:
    print("\nFail to Reject Null Hypothesis: No significant difference in gender distribution across days.")
#Fail to Reject Null Hypothesis: No significant difference in gender distribution across days.