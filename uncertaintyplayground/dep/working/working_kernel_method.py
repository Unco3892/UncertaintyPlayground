import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

# Generate synthetic data
n_samples = 10000
n_features = 10
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1)


# Standardize the feature matrix (optional but recommended)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the Gaussian RBF kernel function
def gaussian_rbf(X, Y, sigma=1):
    return rbf_kernel(X, Y, gamma=1 / (2 * sigma**2))

# Compute the kernel matrix
sigma = 1
kernel_matrix = gaussian_rbf(X, X, sigma=sigma)

# New instance for prediction
# new_instance = np.random.rand(1,n_features)
# scaled_new_instance = scaler.transform(new_instance)
# # True target value for the new instance (assuming you have it)
# y_true = 100


# Compute the similarity between the new instance and the dataset instances
instance_similarities = gaussian_rbf(scaled_new_instance, X, sigma=sigma).flatten()

# Compute the Nadaraya-Watson kernel regression prediction
predicted_value = np.dot(instance_similarities, y) / np.sum(instance_similarities)

# Visualize the density of possible values based on similarities
plt.clf()
plt.hist(y, bins=30, alpha=0.5, density=True, label="All Data")
plt.hist(y, bins=30, weights=instance_similarities, alpha=0.5, density=True, label="Weighted Data")
plt.axvline(predicted_value, color="red", linestyle="--", label="Predicted Value")
plt.axvline(y_true, color="green", linestyle="--", label="Real Value")

plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.title("Density of Possible Values Based on Similarities")
plt.show()

print("Predicted value for the new instance:", predicted_value)
print("Real value for the new instance:", y_true)


