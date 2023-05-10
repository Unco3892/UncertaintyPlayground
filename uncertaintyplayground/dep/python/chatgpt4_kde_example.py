import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.uniform(0, 10, size=(100, 1))
y = 2 * X.squeeze() + 1 + np.random.normal(scale=2, size=100)

# Plot the data
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.show()

from sklearn.linear_model import LinearRegression

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the predictions
plt.scatter(X, y, label="True values")
plt.plot(X, y_pred, color="red", label="Predicted values")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

import seaborn as sns

# Calculate residuals
residuals = y - y_pred

# Plot the KDE of residuals
sns.kdeplot(residuals, bw_method='scott')
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.show()


from scipy.stats import gaussian_kde

# Choose a new observation
X_new = np.array([[5]])

# Predict the outcome for the new observation
y_new_pred = model.predict(X_new)

# Calculate the KDE of the residuals
kde = gaussian_kde(residuals)

# Calculate the 95% confidence interval
alpha = 0.95
residual_quantiles = np.percentile(residuals, [(1-alpha)/2*100, (1+alpha)/2*100])

# Calculate the confidence interval for the new prediction
y_new_conf_interval = y_new_pred + residual_quantiles

print(f"Predicted value: {y_new_pred[0]:.2f}")
print(f"{alpha*100}% confidence interval: ({y_new_conf_interval[0]:.2f}, {y_new_conf_interval[1]:.2f})")


# better for plotting
def calculate_confidence_interval(X_value, model, residuals, alpha=0.95):
    y_pred = model.predict(np.array([[X_value]]))
    residual_quantiles = np.percentile(residuals, [(1-alpha)/2*100, (1+alpha)/2*100])
    y_conf_interval = y_pred + residual_quantiles
    return y_conf_interval

# Create a grid of X values
X_grid = np.linspace(X.min(), X.max(), num=500)

# Calculate the confidence intervals for each X value
y_conf_intervals = np.array([calculate_confidence_interval(x, model, residuals) for x in X_grid])

plt.scatter(X, y, label="True values", alpha=0.5)
plt.plot(X_grid, model.predict(X_grid[:, np.newaxis]), color="red", label="Predicted values")
plt.fill_between(X_grid, y_conf_intervals[:, 0], y_conf_intervals[:, 1], color="gray", alpha=0.3, label="95% Confidence interval")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

