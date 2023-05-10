import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.91, random_state=42)

# Standardize the feature matrix (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Gaussian RBF kernel function
def gaussian_rbf(X, Y, sigma=1):
    return rbf_kernel(X, Y, gamma=1 / (2 * sigma**2))

eps = 1e-8

# Function to calculate Nadaraya-Watson kernel regression predictions
def kernel_regression_predict(X_train, y_train, X_test, sigma=1):
    y_pred = []
    for instance in X_test:
        instance = instance.reshape(1, -1)
        instance_similarities = gaussian_rbf(instance, X_train, sigma=sigma).flatten()
        predicted_value = np.dot(instance_similarities, y_train) / (np.sum(instance_similarities) + eps)
        y_pred.append(predicted_value)
    return np.array(y_pred)

# Define the Gaussian Process kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2))

# Train and evaluate different models
models = {
    "Nadaraya-Watson Kernel Regression": kernel_regression_predict(X_train, y_train, X_test, sigma=1),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gaussian Process Regression": GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=10, random_state=42)
}

for model_name, model in models.items():
    if not isinstance(model, np.ndarray):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        y_pred = model
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse:.4f}")
