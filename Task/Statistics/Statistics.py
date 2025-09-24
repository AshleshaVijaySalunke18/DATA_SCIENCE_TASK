#Section 1: Central Tendency and Spread

import numpy as np
data = [70, 72, 75, 68, 69, 74, 71, 76]
# a. Mean
mean = np.mean(data)
# b. Median
median = np.median(data)
# c. Mean Absolute Deviation (MAD)
mad = np.mean(np.abs(data - mean))
# d. Standard Deviation
std_dev = np.std(data)
print("Mean:", mean)
print("Median:", median)
print("MAD:", mad)
print("Standard Deviation:", std_dev)

'''# Q2. [Interpretation] 
Two students scored the same average marks in two different 
subjects, but the spread of scores is different.
a. What statistical measure can help compare the variability of their scores?
b. Why is MAD insufficient '''

import numpy as np
from statistics import mean
# Define scores for two students (same mean, different spread)
student1 = np.array([70, 70, 70, 70, 70])  # No variability
student2 = np.array([60, 65, 70, 75, 80])  # High variability
# Mean
mean1 = np.mean(student1)
mean2 = np.mean(student2)
# Standard Deviation
std1 = np.std(student1, ddof=0)
std2 = np.std(student2, ddof=0)
# Mean Absolute Deviation (MAD)
mad1 = np.mean(np.abs(student1 - mean1))
mad2 = np.mean(np.abs(student2 - mean2))
# Display results
print("Student 1 - Mean:", mean1, " | SD:", std1, " | MAD:", mad1)
print("Student 2 - Mean:", mean2, " | SD:", std2, " | MAD:", mad2)


#Section 2: Data Transformation and Effec'ts
'''Q3. [Application] 
A dataset has a mean of 60 and standard deviation of 10.
 Each data point is increased by 5.
a. What will be the new mean and standard deviation?
b. Now, multiply each point by 2. What happens to the
 mean and standard deviation?
'''
'''Q4. [Code] 
Simulate the above transformations using Python and numpy
 for data = np.array([60, 65, 55, 70, 50]).'''
 
import numpy as np
# Original data
data = np.array([60, 65, 55, 70, 50])
# Original Mean and SD
original_mean = np.mean(data)
original_std = np.std(data)
# a. Add 5 to each element
data_plus_5 = data + 5
mean_plus_5 = np.mean(data_plus_5)
std_plus_5 = np.std(data_plus_5)
# b. Multiply each point by 2
data_times_2 = data_plus_5 * 2
mean_times_2 = np.mean(data_times_2)
std_times_2 = np.std(data_times_2)
# Print results
print("Original Data:", data)
print("Original Mean:", original_mean)
print("Original SD:", original_std)
print("\nAfter Adding 5:")
print("New Data:", data_plus_5)
print("New Mean:", mean_plus_5)
print("New SD:", std_plus_5)
print("\nAfter Multiplying by 2:")
print("Final Data:", data_times_2)
print("New Mean:", mean_times_2)
print("New SD:", std_times_2)

'''
Section 3: Histogram & Density Curves
Q5. [Visualisation] 
Using matplotlib and seaborn, plot a histogram and its corresponding density
 curve for randomly generated marks of 100 students.
Explain what the peak and spread of the curve indicate.
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate random marks for 100 students (normal distribution)
np.random.seed(42)  # for reproducibility
marks = np.random.normal(loc=70, scale=10, size=100)  # Mean=70, SD=10

# Plot histogram and density curve
plt.figure(figsize=(10, 6))
sns.histplot(marks, kde=True, bins=15, color='skyblue', edgecolor='black')

# Add labels and title
plt.title("Histogram & Density Curve of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency / Density")
plt.grid(True)
plt.show()

'''
Section 4: Normal Distribution & Skewness
Q6. [Theory] 
Define and give examples of:
a. Symmetrical Distribution
b. Left-skewed Distribution
c. Right-skewed Distribution
d. Kurtosis
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# Create distributions
np.random.seed(0)

# a. Symmetrical (Normal Distribution)
sym_data = np.random.normal(loc=0, scale=1, size=1000)

# b. Left-skewed (Using Beta Distribution)
left_skewed = np.random.beta(a=5, b=2, size=1000)

# c. Right-skewed (Using Beta Distribution)
right_skewed = np.random.beta(a=2, b=5, size=1000)

# Plotting all three
plt.figure(figsize=(15, 4))

# Symmetrical
plt.subplot(1, 3, 1)
sns.histplot(sym_data, kde=True, color='blue')
plt.title(f'Symmetrical\nSkew: {skew(sym_data):.2f}, Kurtosis: {kurtosis(sym_data):.2f}')

# Left-skewed
plt.subplot(1, 3, 2)
sns.histplot(left_skewed, kde=True, color='green')
plt.title(f'Left-skewed\nSkew: {skew(left_skewed):.2f}, Kurtosis: {kurtosis(left_skewed):.2f}')

# Right-skewed
plt.subplot(1, 3, 3)
sns.histplot(right_skewed, kde=True, color='red')
plt.title(f'Right-skewed\nSkew: {skew(right_skewed):.2f}, Kurtosis: {kurtosis(right_skewed):.2f}')

plt.tight_layout()
plt.show()

'''
Q7. [Code] 
Generate left-skewed and right-skewed distributions using Python. Plot them using histograms.

'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate Right-Skewed Distribution (e.g., exponential)
right_skewed = np.random.exponential(scale=2.0, size=1000)

# Generate Left-Skewed Distribution by inverting a right-skewed one
left_skewed = -1 * np.random.exponential(scale=2.0, size=1000)

# Create histograms
plt.figure(figsize=(14, 5))

# Left-skewed
plt.subplot(1, 2, 1)
sns.histplot(left_skewed, kde=True, color='tomato', bins=30)
plt.title("Left-Skewed Distribution")
plt.xlabel("Values")
plt.ylabel("Frequency")

# Right-skewed
plt.subplot(1, 2, 2)
sns.histplot(right_skewed, kde=True, color='steelblue', bins=30)
plt.title("Right-Skewed Distribution")
plt.xlabel("Values")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

'''
Section 5: Chebyshev’s Inequality
Q8. [Problem Solving] 
Given a dataset with:
- Mean salary (μ) = ₹40,000
- Standard Deviation (σ) = ₹10,000

Use Chebyshev’s Theorem to find:
a. The minimum % of employees earning between ₹20,000 and ₹60,000.
b. Compare it to the normal distribution assumption (empirical rule).
'''
import numpy as np

# Given values
mean_salary = 40000  # μ
std_salary = 10000   # σ

# Range: ₹20,000 to ₹60,000
lower_bound = 20000
upper_bound = 60000

# a. Calculate k (number of standard deviations from mean)
k = (upper_bound - mean_salary) / std_salary  # or (mean_salary - lower_bound) / std_salary

# Chebyshev's Theorem: P(|X - μ| < kσ) ≥ 1 - 1/k^2
chebyshev_min_prob = 1 - (1 / (k ** 2))

print("a. Using Chebyshev’s Inequality:")
print(f"k = {k}")
print(f"Minimum percentage of employees earning between ₹20,000 and ₹60,000: {chebyshev_min_prob * 100:.2f}%")

# b. Empirical Rule (Normal Distribution Assumption)
# For normal distribution, within ±2σ ≈ 95%
empirical_rule_percent = 95

print("\nb. Using Empirical Rule (Assumes Normal Distribution):")
print(f"Approximate percentage: {empirical_rule_percent}%")
print("\n Comparison:")
print(f"Chebyshev (any distribution): ≥ {chebyshev_min_prob * 100:.2f}%")
print(f"Empirical Rule (normal distribution): ≈ {empirical_rule_percent}%")

'''
Section 6: Log Transformation in ML
Q9. [Python Application] 
You are given a highly skewed income dataset. Apply np.log() transformation and plot the result.
Discuss how the transformation helps in ML models.

'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a highly skewed income dataset using exponential distribution
np.random.seed(42)
income = np.random.exponential(scale=50000, size=1000)

# Apply log transformation (add 1 to avoid log(0))
log_income = np.log(income + 1)

# Plot original vs. log-transformed distributions
plt.figure(figsize=(14, 5))

# Original Skewed Income
plt.subplot(1, 2, 1)
sns.histplot(income, kde=True, color='orange', bins=30)
plt.title("Original Skewed Income")
plt.xlabel("Income")
plt.ylabel("Frequency")

# Log-Transformed Income
plt.subplot(1, 2, 2)
sns.histplot(log_income, kde=True, color='seagreen', bins=30)
plt.title("Log-Transformed Income")
plt.xlabel("Log(Income + 1)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

'''
Section 7: SciPy Applications
Q10. [SciPy Hands-on] 
Write Python code using SciPy to:
'''
#a. Compute integral of f(x) = x^2 from 0 to 1

from scipy import integrate

# Define the function f(x) = x^2
f = lambda x: x**2

# Integrate from 0 to 1
result, error = integrate.quad(f, 0, 1)

print("a. Integral of x^2 from 0 to 1:", result)

#b. Interpolate given data points
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

# Given data points (x, y)
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])  # y = x^2

# Interpolate using quadratic curve
f_interp = interpolate.interp1d(x, y, kind='quadratic')

# Generate smooth x values and compute interpolated y values
x_new = np.linspace(0, 4, 100)
y_new = f_interp(x_new)

# Plot original and interpolated curve
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'o', label='Original Points')
plt.plot(x_new, y_new, '-', label='Interpolated Curve')
plt.title("b. Interpolation Example")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

#c. Perform basic image transformation 
from scipy import ndimage
from scipy.datasets import ascent  # For sample image (SciPy ≥1.11)
import matplotlib.pyplot as plt

# Load a sample grayscale image
image = ascent()

# Image transformations
rotated = ndimage.rotate(image, 45)                 # Rotate by 45 degrees
zoomed = ndimage.zoom(image, 0.5)                   # Zoom out to 50%
blurred = ndimage.gaussian_filter(image, sigma=3)   # Gaussian blur

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(rotated, cmap='gray')                                                                                
plt.title("Rotated 45°")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(zoomed, cmap='gray')
plt.title("Zoomed (0.5x)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(blurred, cmap='gray')
plt.title("Gaussian Blurred (σ=3)")
plt.axis('off')

plt.tight_layout()
plt.show()
