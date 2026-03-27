# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:43:34 2026

@author: Ashlesha
"""

"""

Topic: Forecasting – Time Series

## 1. Business Problem

### 1.1 Business Objective

The main objective of this project is to forecast future values (such as sales, demand, or revenue) using historical time series data. Accurate forecasting helps organizations make informed decisions regarding inventory management, production planning, budgeting, and resource allocation.

### 1.2 Constraints

* Missing or incomplete data
* Limited historical records
* Presence of noise and outliers
* Complex trends and seasonality
* External influencing factors (holidays, economic conditions)

---

## 2. Data Dictionary

| Feature Name | Data Type   | Description                    | Relevance                    |
| ------------ | ----------- | ------------------------------ | ---------------------------- |
| Date         | DateTime    | Represents time of observation | Highly relevant (time index) |
| Sales        | Numeric     | Target variable to forecast    | Highly relevant              |
| Month        | Numeric     | Extracted from date            | Useful for seasonality       |
| Year         | Numeric     | Extracted from date            | Helps identify trend         |
| Day          | Numeric     | Day-level granularity          | Optional                     |
| Holiday      | Binary      | Indicates special events       | Useful                       |
| Store/Region | Categorical | Location identifier            | Depends on dataset           |

---

## 3. Data Pre-processing

### 3.1 Data Cleaning & Feature Engineering

* Converted date column into datetime format
* Sorted dataset based on time
* Handled missing values using forward fill or interpolation
* Created new features like month, year, and lag variables
* Generated rolling averages for smoothing

### 3.2 Outlier Treatment

* Identified outliers using boxplots and statistical methods
* Treated outliers by capping or smoothing
* Removed extreme anomalies when necessary

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Summary

* Calculated mean, median, and standard deviation
* Checked distribution of data
* Identified missing values

### 4.2 Trend Analysis

* Visualized time vs sales data
* Identified upward/downward trend patterns

### 4.3 Seasonality Analysis

* Observed repeating patterns over time
* Identified monthly/weekly seasonal effects

---

## 5. Model Building

### 5.1 Moving Average Method

* Used past observations to smooth data
* Suitable for short-term forecasting

### 5.2 Exponential Smoothing

* Assigned higher weights to recent observations
* Applied:

  * Simple Exponential Smoothing
  * Holt’s Linear Trend Method
  * Holt-Winters Method

### 5.3 Model-Based Approach

* Built regression model using time as independent variable
* Included trend and seasonality components

### 5.4 ARIMA Model

* Applied ARIMA (p, d, q) model
* Checked stationarity using statistical tests
* Used differencing to stabilize data
* Selected optimal parameters
* Generated forecasts

### 5.5 Model Evaluation

* Evaluated models using:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * MAPE (Mean Absolute Percentage Error)
* Selected best-performing model based on lowest error

---

## 6. Results and Interpretation

* Compared different forecasting models
* Identified the best model based on accuracy
* Analyzed forecast trends and patterns
* Validated predictions with actual data

---

## 7. Business Impact

The forecasting solution provides the following benefits:

* Improves demand prediction accuracy
* Reduces inventory and operational costs
* Enhances supply chain efficiency
* Supports better business decision-making
* Increases profitability
* Enables proactive planning


"""
"""

Problem Statement: -

1.	The dataset consists of monthly totals of international airline passengers from 1995 to 2002. Our main aim is to predict the number of passengers for the next five years using time series forecasting. Prepare a document for each model explaining how many dummy variables you have created and also include the RMSE value for each model.

File: - Airlines.xlsx"""


# =========================================
# AIRLINES TIME SERIES FORECASTING (ALL-IN-ONE)
# =========================================

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

# STEP 2: Load Dataset
data = pd.read_excel("C:/Data-Science/Task/Time Series/Airlines Data.xlsx")
# STEP 3: Data Preprocessing
data["Month"] = pd.to_datetime(data["Month"])

# Create time index
data["t"] = np.arange(1, len(data) + 1)

# Square term (for quadratic model)
data["t_square"] = data["t"] ** 2

# Log transformation
data["log_passengers"] = np.log(data["Passengers"])

# Extract month and create dummy variables
data["month"] = data["Month"].dt.strftime("%b")
dummies = pd.get_dummies(data["month"])
'''
	Apr	Aug	Dec	Feb	Jan	Jul	Jun	Mar	May	Nov	Oct	Sep
0	False	False	False	False	True	False	False	False	False	False	False	False
1	False	False	False	True	False	False	False	False	False	False	False	False
2	False	False	False	False	False	False	False	True	False	False	False	False
3	True	False	False	False	False	False	False	False	False	False	False	False
4	False	False	False	False	False	False	False	False	True	False	False	False
5	False	False	False	False	False	False	True	False	False	False	False	False
.
.
.
'''
# Combine dataset with dummies
data = pd.concat([data, dummies], axis=1)

# STEP 4: Train-Test Split (Last 12 months for testing)
train = data.head(len(data) - 12)
test = data.tail(12)

# STEP 5: RMSE Function
def rmse(pred, actual):
    return np.sqrt(mean_squared_error(pred, actual))

# =========================================
# MODEL BUILDING
# =========================================

# 1. Linear Model
model_linear = smf.ols("Passengers ~ t", data=train).fit()
pred_linear = model_linear.predict(test["t"])
rmse_linear = rmse(pred_linear, test["Passengers"])

# 2. Exponential Model
model_exp = smf.ols("log_passengers ~ t", data=train).fit()
pred_exp = np.exp(model_exp.predict(test["t"]))
rmse_exp = rmse(pred_exp, test["Passengers"])

# 3. Quadratic Model
model_quad = smf.ols("Passengers ~ t + t_square", data=train).fit()
pred_quad = model_quad.predict(test[["t", "t_square"]])
rmse_quad = rmse(pred_quad, test["Passengers"])

# 4. Additive Seasonality
model_add_sea = smf.ols(
    "Passengers ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec",
    data=train
).fit()
pred_add_sea = model_add_sea.predict(test)
rmse_add_sea = rmse(pred_add_sea, test["Passengers"])

# 5. Additive Seasonality + Linear
model_add_sea_lin = smf.ols(
    "Passengers ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec",
    data=train
).fit()
pred_add_sea_lin = model_add_sea_lin.predict(test)
rmse_add_sea_lin = rmse(pred_add_sea_lin, test["Passengers"])

# 6. Additive Seasonality + Quadratic
model_add_sea_quad = smf.ols(
    "Passengers ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec",
    data=train
).fit()
pred_add_sea_quad = model_add_sea_quad.predict(test)
rmse_add_sea_quad = rmse(pred_add_sea_quad, test["Passengers"])

# 7. Multiplicative Seasonality
model_mul_sea = smf.ols(
    "log_passengers ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec",
    data=train
).fit()
pred_mul_sea = np.exp(model_mul_sea.predict(test))
rmse_mul_sea = rmse(pred_mul_sea, test["Passengers"])

# 8. Multiplicative Seasonality + Linear
model_mul_sea_lin = smf.ols(
    "log_passengers ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec",
    data=train
).fit()
pred_mul_sea_lin = np.exp(model_mul_sea_lin.predict(test))
rmse_mul_sea_lin = rmse(pred_mul_sea_lin, test["Passengers"])

# =========================================
# RESULTS COMPARISON
# =========================================

results = pd.DataFrame({
    "Model": [
        "Linear",
        "Exponential",
        "Quadratic",
        "Additive Seasonality",
        "Additive Seasonality + Linear",
        "Additive Seasonality + Quadratic",
        "Multiplicative Seasonality",
        "Multiplicative Seasonality + Linear"
    ],
    "RMSE": [
        rmse_linear,
        rmse_exp,
        rmse_quad,
        rmse_add_sea,
        rmse_add_sea_lin,
        rmse_add_sea_quad,
        rmse_mul_sea,
        rmse_mul_sea_lin
    ],
    "Dummy Variables": [
        0, 0, 0, 12, 12, 12, 12, 12
    ]
})

print("\n===== MODEL COMPARISON =====")
print(results)

'''
                               Model        RMSE            Dummy Variables
0                               Linear   53.199237                0
1                          Exponential   46.057361                0
2                            Quadratic   48.051889                0
3                 Additive Seasonality  132.819785               12
4        Additive Seasonality + Linear   35.348957               12
5     Additive Seasonality + Quadratic   26.360818               12
6           Multiplicative Seasonality  140.063202               12
7  Multiplicative Seasonality + Linear   10.519173               12

'''
# =========================================
# BEST MODEL
# =========================================

best_model = results.loc[results["RMSE"].idxmin()]

print("\n===== BEST MODEL =====")
print(best_model)

'''
Model              Multiplicative Seasonality + Linear
RMSE                                         10.519173
Dummy Variables                                     12
'''

"""
Problem Statement: -

2.	The dataset consists of quarterly sales data of Coca-Cola from 1986 to 1996. Predict sales for the next two years by using time series forecasting and prepare a document for each model explaining how many dummy variables you have created and also include the RMSE value for each model.

File:- CocaCola_Sales_RawData.xlsx"""

import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_excel("C:/Data-Science/Task/Time Series/CocaCola_Sales_Rawdata.xlsx")


# Create time features
data["t"] = np.arange(1,len(data)+1)
data["log_sales"] = np.log(data["Sales"])

# Create quarter dummies
data["Quarter"] = data["Quarter"].str[0:2]
d = pd.get_dummies(data["Quarter"]); data = pd.concat([data,d],axis=1)

# Train-test split
train, test = data[:-4], data[-4:]

# Build model (Additive + Trend)
m = smf.ols("Sales ~ t + Q1+Q2+Q3+Q4", data=train).fit()

# Prediction & RMSE
pred = m.predict(test)
rmse = np.sqrt(mean_squared_error(test["Sales"], pred))

print("RMSE:", rmse, "| Dummy:", 4)


'''RMSE: 464.98289300269244 | Dummy: 4'''





"""

Problem Statement: - 

A plastics manufacturing plant has recorded their monthly sales data from 1949 to 1953. Perform forecasting on the data and bring out insights from it and forecast the sale for the next year. 

Plastic Sales.csv"""

# Import libraries
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("C:/Data-Science/Task/Time Series/PlasticSales.csv")

# Convert date & create features
data["Month"] = pd.to_datetime(data["Month"])
data["t"] = np.arange(1,len(data)+1)

# Create month dummies
data["m"] = data["Month"].dt.strftime("%b")
d = pd.get_dummies(data["m"]); data = pd.concat([data,d],axis=1)

# Split data
train, test = data[:-12], data[-12:]

# Model (Seasonality)
m = smf.ols("Sales ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec", data=train).fit()

# Predict & RMSE
pred = m.predict(test)
rmse = np.sqrt(mean_squared_error(test["Sales"], pred))

print("RMSE:", rmse, "| Dummy:", 12)


'''RMSE: 235.60267356646557 | Dummy: 12'''




"""

Problem Statement: -

Solar power consumption has been recorded by city councils at regular intervals. The reason behind doing so is to understand how businesses are using solar power so that they can cut down on nonrenewable sources of energy and shift towards renewable energy. Based on the data, build a forecasting model and provide insights on it. 

Solarpower.csv

"""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:/Data-Science/Task/Time Series/solarpower.csv")
data.columns = data.columns.str.strip()
data = data.rename(columns={data.columns[1]: "Consumption"})

data["t"] = np.arange(1, len(data)+1)

# Fix month issue
data.iloc[:,0] = pd.to_datetime(data.iloc[:,0])
data["month"] = data.iloc[:,0].dt.month

dummies = pd.get_dummies(data["month"], drop_first=True)
data = pd.concat([data, dummies], axis=1)

train = data[:int(len(data)*0.8)].copy()
test = data[int(len(data)*0.8):].copy()

train["log_y"] = np.log(train["Consumption"])

model = smf.ols("log_y ~ t + " + " + ".join(map(str, dummies.columns)), data=train).fit()

pred = np.exp(model.predict(test))

rmse = np.sqrt(mean_squared_error(test["Consumption"], pred))
print("RMSE:", rmse)
'''RMSE: 235.60267356646557'''