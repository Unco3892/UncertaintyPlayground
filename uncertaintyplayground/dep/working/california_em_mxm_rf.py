from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Introduce missing values (e.g., 20% missing values)
# missing_rate = 0.2
# mask = np.random.binomial(1, 1 - missing_rate, X.shape).astype(bool)
# X_missing = X.copy()
# X_missing[mask] = np.nan
# Introduce missing values (e.g., 20% missing values)
missing_rate = 0.2
mask = np.random.binomial(1, 1 - missing_rate, X.size).astype(bool)
X_missing = X.copy()
X_missing[np.unravel_index(np.random.choice(X.size, int(missing_rate * X.size), replace=False), X.shape)] = np.nan

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_missing, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Scenario 1: Mean imputation
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

reg1 = LinearRegression()
reg1.fit(X_train_imputed, y_train)

y_pred_train1 = reg1.predict(X_train_imputed)
y_pred_test1 = reg1.predict(X_test_imputed)

mse_train1 = mean_squared_error(y_train, y_pred_train1)
mse_test1 = mean_squared_error(y_test, y_pred_test1)

# Scenario 2: GMM-based imputation
n_components = 10
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X_train_imputed)

responsibilities_train = gmm.predict_proba(X_train_imputed)
responsibilities_test = gmm.predict_proba(X_test_imputed)

for i in range(X_train.shape[0]):
    for j in range(X_train.shape[1]):
        if np.isnan(X_train[i, j]):
            X_train[i, j] = np.dot(responsibilities_train[i], [gmm.means_[k, j] for k in range(n_components)])

for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if np.isnan(X_test[i, j]):
            X_test[i, j] = np.dot(responsibilities_test[i], [gmm.means_[k, j] for k in range(n_components)])

reg2 = LinearRegression()
reg2.fit(X_train, y_train)

y_pred_train2 = reg2.predict(X_train)
y_pred_test2 = reg2.predict(X_test)

mse_train2 = mean_squared_error(y_train, y_pred_train2)
mse_test2 = mean_squared_error(y_test, y_pred_test2)

# Scenario 3: GMM-based feature engineering
X_train_extended = np.hstack((X_train_imputed, responsibilities_train))
X_test_extended = np.hstack((X_test_imputed, responsibilities_test))

reg3 = LinearRegression()
reg3.fit(X_train_extended, y_train)

y_pred_train3 = reg3.predict(X_train_extended)
y_pred_test3 = reg3.predict(X_test_extended)

mse_train3 = mean_squared_error(y_train, y_pred_train3)
mse_test3 = mean_squared_error(y_test,y_pred_test3)

# Compare the results
print("Scenario 1: Mean Imputation")
print("Training Mean Squared Error: ", mse_train1)
print("Testing Mean Squared Error: ", mse_test1)

print("\nScenario 2: GMM-based Imputation")
print("Training Mean Squared Error: ", mse_train2)
print("Testing Mean Squared Error: ", mse_test2)

print("\nScenario 3: GMM-based Feature Engineering")
print("Training Mean Squared Error: ", mse_train3)
print("Testing Mean Squared Error: ", mse_test3)

# 1. Mean imputation, which fills the missing values with the mean of the respective feature.
# 2. GMM-based imputation, which estimates the missing values using the Gaussian Mixture Model approach.
# 3. GMM-based feature engineering, which uses the responsibilities calculated by the GMM as new features.

#-----------------------------------
from sklearn.ensemble import RandomForestRegressor

# Remove rows with missing values
mask_complete = np.isnan(X_missing).any(axis=1)
X_complete = X[~mask_complete]
y_complete = y[~mask_complete]

# Split the dataset into training and testing sets
X_train_complete, X_test_complete, y_train_complete, y_test_complete = train_test_split(X_complete, y_complete, test_size=0.2, random_state=42)

# Standardize the data
scaler_complete = StandardScaler()
X_train_complete = scaler_complete.fit_transform(X_train_complete)
X_test_complete = scaler_complete.transform(X_test_complete)

# Linear Regression with complete data
reg_complete = LinearRegression()
reg_complete.fit(X_train_complete, y_train_complete)

y_pred_train_complete = reg_complete.predict(X_train_complete)
y_pred_test_complete = reg_complete.predict(X_test_complete)

mse_train_complete = mean_squared_error(y_train_complete, y_pred_train_complete)
mse_test_complete = mean_squared_error(y_test_complete, y_pred_test_complete)

# Random Forest with complete data
rf_complete = RandomForestRegressor(n_estimators=100, random_state=42)
rf_complete.fit(X_train_complete, y_train_complete)

y_pred_train_rf_complete = rf_complete.predict(X_train_complete)
y_pred_test_rf_complete = rf_complete.predict(X_test_complete)

mse_train_rf_complete = mean_squared_error(y_train_complete, y_pred_train_rf_complete)
mse_test_rf_complete = mean_squared_error(y_test_complete, y_pred_test_rf_complete)

#-------------------------------------------------------------------------------------------

print("-----")
# Compare the results
print("Scenario 0a: Linear Regression with Complete Data")
print("Training Mean Squared Error: ", mse_train_complete)
print("Testing Mean Squared Error: ", mse_test_complete)

print("\nScenario 0b: Random Forest with Complete Data")
print("Training Mean Squared Error: ", mse_train_rf_complete)
print("Testing Mean Squared Error: ", mse_test_rf_complete)

print("\nScenario 1: Mean Imputation")
print("Training Mean Squared Error: ", mse_train1)
print("Testing Mean Squared Error: ", mse_test1)

print("\nScenario 2: GMM-based Imputation")
print("Training Mean Squared Error: ", mse_train2)
print("Testing Mean Squared Error: ", mse_test2)

print("\nScenario 3: GMM-based Feature Engineering")
print("Training Mean Squared Error: ", mse_train3)
print("Testing Mean Squared Error: ", mse_test3)

#-------------------------------------------------------------------------------------------
print("-----")
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# MissForest Imputation
missforest_imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42), random_state=42)
X_train_missforest = missforest_imputer.fit_transform(X_train)
X_test_missforest = missforest_imputer.transform(X_test)

reg_missforest = LinearRegression()
reg_missforest.fit(X_train_missforest, y_train)

y_pred_train_missforest = reg_missforest.predict(X_train_missforest)
y_pred_test_missforest = reg_missforest.predict(X_test_missforest)

mse_train_missforest = mean_squared_error(y_train, y_pred_train_missforest)
mse_test_missforest = mean_squared_error(y_test, y_pred_test_missforest)

# Compare the results
print("Scenario 0a: Linear Regression with Complete Data")
print("Training Mean Squared Error: ", mse_train_complete)
print("Testing Mean Squared Error: ", mse_test_complete)

print("\nScenario 0b: Random Forest with Complete Data")
print("Training Mean Squared Error: ", mse_train_rf_complete)
print("Testing Mean Squared Error: ", mse_test_rf_complete)

print("\nScenario 1: Mean Imputation")
print("Training Mean Squared Error: ", mse_train1)
print("Testing Mean Squared Error: ", mse_test1)

print("\nScenario 2: GMM-based Imputation")
print("Training Mean Squared Error: ", mse_train2)
print("Testing Mean Squared Error: ", mse_test2)

print("\nScenario 3: GMM-based Feature Engineering")
print("Training Mean Squared Error: ", mse_train3)
print("Testing Mean Squared Error: ", mse_test3)

print("\nScenario 4: MissForest Imputation")
print("Training Mean Squared Error: ", mse_train_missforest)
print("Testing Mean Squared Error: ", mse_test_missforest)
