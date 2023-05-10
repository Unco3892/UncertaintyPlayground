import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_and_predict_gp_regression(X_train, y_train, X_test):
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the kernel for the GaussianProcessRegressor
    kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)

    # Fit the GaussianProcessRegressor model
    gpr_model = GaussianProcessRegressor(kernel=kernel)
    gpr_model.fit(X_train, y_train)

    # Make predictions for the test set
    y_pred = gpr_model.predict(X_test)

    return y_pred
