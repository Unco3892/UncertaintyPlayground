import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Introduce some missing values (e.g., 20% missing values) for demonstration purposes
np.random.seed(42)
missing_rate = 0.2
mask = np.random.binomial(1, 1 - missing_rate, X.shape).astype(bool)
X_missing = X.copy()
X_missing[mask] = np.nan

# Impute missing values using the mean (as a baseline for comparison)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_missing)

# Apply Gaussian Mixture Model to the dataset
n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X_imputed)

# Perform E-step: compute responsibilities
responsibilities = gmm.predict_proba(X_imputed)

# Perform M-step: update the means, covariances, and weights of the Gaussian components
for i in range(n_components):
    resp = responsibilities[:, i].reshape(-1, 1)
    gmm.means_[i] = np.nanmean(resp * X_missing, axis=0) / resp.mean()
    diff = X_missing - gmm.means_[i]
    centered_diff = np.where(np.isnan(diff), 0, diff)
    gmm.covariances_[i] = np.dot((resp * centered_diff).T, centered_diff) / resp.sum()

gmm.weights_ = responsibilities.mean(axis=0)

# Estimate missing values using the Gaussian Mixture Model
for i in range(X_missing.shape[0]):
    for j in range(X_missing.shape[1]):
        if np.isnan(X_missing[i, j]):
            X_missing[i, j] = np.dot(responsibilities[i], [gmm.means_[k, j] for k in range(n_components)])

# Compare the original data, imputed data, and the data with estimated missing values
X_original = pd.DataFrame(X, columns=feature_names)
X_imputed = pd.DataFrame(X_imputed, columns=feature_names)
X_estimated = pd.DataFrame(X_missing, columns=feature_names)

print("Original data:\n", X_original.head())
print("\nImputed data (mean imputation):\n", X_imputed.head())
print("\nData with estimated missing values (GMM):\n", X_estimated.head())
