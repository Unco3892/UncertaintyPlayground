import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize the feature matrix (optional but recommended)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the Gaussian RBF kernel function
def gaussian_rbf(X, Y, sigma=1):
    return rbf_kernel(X, Y, gamma=1 / (2 * sigma**2))

# Compute the kernel matrix
sigma = 1
kernel_matrix = gaussian_rbf(X, X, sigma=sigma)

# Select a few existing instances for prediction
selected_indices = [100, 1000, 5000, 10000, 15000]

# Perform predictions and visualize the density of possible values for the selected instances
for idx in selected_indices:
    instance = X[idx].reshape(1, -1)
    y_true = y[idx]
    
    # Compute the similarity between the selected instance and the dataset instances
    instance_similarities = gaussian_rbf(instance, X, sigma=sigma).flatten()

    # Compute the Nadaraya-Watson kernel regression prediction
    predicted_value = np.dot(instance_similarities, y) / np.sum(instance_similarities)

    # Visualize the density of possible values based on similarities
    plt.hist(y, bins=30, alpha=0.5, density=True, label="All Data")
    plt.hist(y, bins=30, weights=instance_similarities, alpha=0.5, density=True, label="Weighted Data")
    plt.axvline(predicted_value, color="red", linestyle="--", label="Predicted Value")
    plt.axvline(y_true, color="green", linestyle="--", label="Real Value")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Density of Possible Values for Instance {idx}")
    plt.show()

    print(f"Predicted value for instance {idx}:", predicted_value)
    print(f"Real value for instance {idx}:", y_true)

