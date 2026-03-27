# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 14:01:15 2025

@author: Ashlesha
"""

#1. Mean, Median, MAD & Standard Deviation – Real World
'''
Mean, Median, MAD & Standard Deviation – Real World
Problem:
You are analyzing monthly sales data for two shops.
shop1_sales = [2200, 2250, 2300, 2350, 2400, 4000]
shop2_sales = [2000, 2100, 2300, 2500, 2700, 2800]
•	a. Calculate the mean, median, MAD, and standard deviation for both shops.
•	b. Which shop shows higher consistency in sales and why?
•	c. Suppose the outlier in shop1 is corrected to 2450. How do the metrics change?

'''
import numpy as np
from scipy import stats

# Monthly sales of two shops
shop1_sales = [2200, 2250, 2300, 2350, 2400, 4000]
shop2_sales = [2000, 2100, 2300, 2500, 2700, 2800]

# Function to calculate statistics
def describe(data):
    mean = np.mean(data)                             # Average
    median = np.median(data)                         # Middle value
    mad = np.mean(np.abs(data - mean))               # Mean Absolute Deviation
    std = np.std(data, ddof=0)                       # Standard Deviation (Population)
    return mean, median, mad, std

# Display stats for both shops
print("Shop 1:", describe(shop1_sales))
print("Shop 2:", describe(shop2_sales))

# Correcting outlier in Shop 1 (4000 → 2450)
shop1_corrected = [2200, 2250, 2300, 2350, 2400, 2450]
print("Shop 1 (Corrected):", describe(shop1_corrected))

#2. Effect of Data Transformation on Spread
'''
. Effect of Data Transformation on Spread
Problem:
Given the dataset:
data = [25, 30, 35, 40, 45, 50]
•	a. Compute the mean and standard deviation.
•	b. Now apply:
o	Addition: Add 5 to each value
o	Multiplication: Multiply each value by 2
o	Log transformation: Apply np.log(data)
•	c. Discuss the effect of each transformation on center and spread.
________________________________________

'''
data = np.array([25, 30, 35, 40, 45, 50])

# Original metrics
print("Original: ", np.mean(data), np.std(data))

# Add 5 to each value
add5 = data + 5
print("Add 5: ", np.mean(add5), np.std(add5))  # Mean increases, std unchanged

# Multiply by 2
mul2 = data * 2
print("Multiply by 2: ", np.mean(mul2), np.std(mul2))  # Both mean and std double

# Apply log transformation
log_data = np.log(data)
print("Log: ", np.mean(log_data), np.std(log_data))  # Mean decreases, std compressed

# 3. Density Curve vs Histogram
'''
Problem:
Generate 1000 height values assuming a normal distribution (mean=160 cm, std=10 cm).
•	a. Plot histogram and KDE using seaborn.
•	b. Manually create bins (intervals of 5 cm) and compute relative frequency.
•	c. Approximate the area under the density curve between 150–170 cm. What does it represent?

'''
import seaborn as sns
import matplotlib.pyplot as plt

# Generate synthetic height data
heights = np.random.normal(loc=160, scale=10, size=1000)

# Plot histogram and KDE (Kernel Density Estimate)
sns.histplot(heights, kde=True, bins=30)
plt.title("Height Distribution")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.show()

# Compute relative frequency using manual bins (5 cm interval)
bins = np.arange(140, 181, 5)
counts, _ = np.histogram(heights, bins=bins)
rel_freq = counts / counts.sum()
print("Relative Frequency (each bin):", rel_freq)

# Estimate proportion of heights between 150 and 170 cm
within_range = np.sum((heights >= 150) & (heights <= 170)) / len(heights)
print("Proportion between 150–170 cm:", within_range)


# 4. Skewness & Kurtosis Comparison
'''
Problem:
Generate three datasets:
•	A left-skewed dataset
•	A right-skewed dataset
•	A symmetric dataset (normal)
For each:
•	a. Plot histogram and calculate skewness and kurtosis using scipy.stats
•	b. Interpret the shape and tail behavior

'''
from scipy.stats import skew, kurtosis

# Generate left-skewed, right-skewed, and symmetric datasets
left_skewed = np.random.beta(5, 2, size=1000)
right_skewed = np.random.beta(2, 5, size=1000)
symmetric = np.random.normal(0, 1, size=1000)

datasets = {'Left Skewed': left_skewed, 'Right Skewed': right_skewed, 'Symmetric': symmetric}

# Loop through datasets and visualize calculate skewness and kurtosis
for name, data in datasets.items():
    sns.histplot(data, kde=True)
    plt.title(" Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    
    print(f"{name}  Skewness: {skew(data):.2f}, Kurtosis: {kurtosis(data):.2f}")
    
    
#5. Chebyshev's Inequality on Unknown Distribution
'''
Problem:
A dataset has mean income = ₹50,000 and standard deviation = ₹12,000.
•	a. Without knowing the distribution, use Chebyshev’s theorem to estimate what percentage of individuals earn between ₹26,000 and ₹74,000.
•	b. Compare this with the Empirical Rule (assuming normal distribution).

'''
mean_income = 50000
std_dev = 12000

# Calculate Chebyshev's bound for values between ₹26,000 and ₹74,000
k = (74000 - 50000) / std_dev  # k = 2
chebyshev = 1 - 1 / k**2
print("Chebyshev’s Estimate:", chebyshev * 100, "% of data lies between ₹26k–₹74k")

# Compare with empirical rule (normal distribution)
print("Empirical Rule (Normal Distribution): ~95% within 2 standard deviations")

#6. Real-World Log Transformation
'''
Problem:
Load a CSV file (or generate synthetic) containing:
•	Population sizes of 1000 cities
•	Income distribution of households
Tasks:
•	a. Plot histograms for both features
•	b. Apply log transformation and re-plot
•	c. Explain how log helps in compressing skewed data
•	d. Comment on interpretability after transformation

'''


# Generate synthetic log-normal data for population and income
pop = np.random.lognormal(mean=10, sigma=1, size=1000)
income = np.random.lognormal(mean=10, sigma=1.5, size=1000)

# a. Plot histograms before transformation
sns.histplot(pop, bins=30, kde=True)
plt.title("Population Distribution")
plt.show()

sns.histplot(income, bins=30, kde=True)
plt.title("Income Distribution")
plt.show()

# b. Apply log transformation and plot
sns.histplot(np.log(pop), bins=30, kde=True)
plt.title("Log-Transformed Population")
plt.show()

sns.histplot(np.log(income), bins=30, kde=True)
plt.title("Log-Transformed Income")
plt.show()


#7. SciPy Applications – Linear Algebra and Interpolation
'''
Problem:
•	a. Create a 3x3 matrix and compute its determinant using scipy.linalg.det()
•	b. Use scipy.interpolate.interp1d() to interpolate the following points and estimate y at x = 3.5:
x = [1, 2, 4, 5]
y = [1, 4, 2, 5]

'''
from scipy import linalg, interpolate

# a. Create 3x3 matrix and compute determinant
A = np.array([[2, 3, 1], [4, 1, 5], [7, 2, 6]])
det_A = linalg.det(A)
print("Determinant of matrix A:", det_A)

# b. Linear interpolation
x = [1, 2, 4, 5]
y = [1, 4, 2, 5]
f = interpolate.interp1d(x, y)

# Estimate y at x = 3.5
print("Interpolated value at x=3.5:", f(3.5))