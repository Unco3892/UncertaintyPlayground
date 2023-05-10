import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm


# Load the California Housing dataset
california = fetch_california_housing()

N = 5000
# Use the first four predictors and the outcome variable 'MedHouseVal'
X = california.data[:N, :4]
y = california.target[:N]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Kernel Ridge Regression model using a Gaussian kernel
krr_model = KernelRidge(kernel='rbf', gamma=0.1, alpha=0.01)
krr_model.fit(X_train, y_train)

# K-Fold cross-validation for model uncertainty estimation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_errors = -cross_val_score(krr_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

# Calculate the standard deviation of the prediction errors
prediction_error_sd = np.sqrt(np.mean(cv_errors))

# Make point predictions for the test set
y_pred = krr_model.predict(X_test)

# Calculate 95% prediction intervals
z_score = norm.ppf(0.975)
lower_bound = y_pred - z_score * prediction_error_sd
upper_bound = y_pred + z_score * prediction_error_sd

# Combine the point predictions and prediction intervals
predictions = np.column_stack((y_pred, lower_bound, upper_bound))

# Print the results
print(predictions)

plt.figure(figsize=(10, 5))
x = np.arange(len(y_test))
plt.plot(x, y_test, 'o', label='Ground Truth', markersize=3)
plt.plot(x, y_pred, 'r.', label='Point Prediction', markersize=3)
plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.5, label='95% Confidence Interval')
plt.legend()
plt.xlabel('Test Data Index')
plt.ylabel('House Value')
plt.title('Kernel Ridge Regression Predictions with 95% Confidence Intervals')
plt.show()

# only for 100 observations
# Plot the point predictions, 95% confidence intervals, and ground truth values for the first 100 test observations
plt.figure(figsize=(10, 5))
x = np.arange(100)
plt.plot(x, y_test[:100], 'o', label='Ground Truth', markersize=3)
plt.plot(x, y_pred[:100], 'r.', label='Point Prediction', markersize=3)
plt.fill_between(x, lower_bound[:100], upper_bound[:100], color='gray', alpha=0.5, label='95% Confidence Interval')
plt.legend()
plt.xlabel('Test Data Index')
plt.ylabel('House Value')
plt.title('Kernel Ridge Regression Predictions with 95% Confidence Intervals (First 100 Observations)')
plt.show()
