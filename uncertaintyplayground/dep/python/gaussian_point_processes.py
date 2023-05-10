import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Load the California Housing dataset
california = fetch_california_housing()

# Use the first four predictors and the outcome variable 'MedHouseVal'
X = california.data[:1000, :4]
y = california.target[:1000]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the kernel for the GaussianProcessRegressor
kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)

# Fit the GaussianProcessRegressor model
gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
gpr_model.fit(X_train, y_train)

# Make point predictions and estimate prediction variance for the test set
y_pred, y_pred_var = gpr_model.predict(X_test, return_std=True)

# Calculate 95% prediction intervals
z_score = norm.ppf(0.975)
lower_bound = y_pred - z_score * np.sqrt(y_pred_var)
upper_bound = y_pred + z_score * np.sqrt(y_pred_var)

# Plot the point predictions, 95% confidence intervals, and ground truth values for the first 100 test observations
plt.figure(figsize=(10, 5))
x = np.arange(100)
plt.plot(x, y_test[:100], 'o', label='Ground Truth', markersize=3)
plt.plot(x, y_pred[:100], 'r.', label='Point Prediction', markersize=3)
plt.fill_between(x, lower_bound[:100], upper_bound[:100], color='gray', alpha=0.5, label='95% Confidence Interval')
plt.legend()
plt.xlabel('Test Data Index')
plt.ylabel('House Value')
plt.title('Gaussian Process Regression Predictions with 95% Confidence Intervals (First 100 Observations)')
plt.show()

