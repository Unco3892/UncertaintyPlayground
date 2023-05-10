from lazypredict.Supervised import LazyRegressor
import numpy as np
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

chosen_regressors  = ['AdaBoostRegressor', 'BaggingRegressor', 'BayesianRidge', 'DecisionTreeRegressor', 'DummyRegressor', 'ElasticNet', 'ElasticNetCV', 'ExtraTreeRegressor', 'ExtraTreesRegressor', 'GammaRegressor', 'GradientBoostingRegressor', 'HistGradientBoostingRegressor', 'HuberRegressor', 'KNeighborsRegressor', 'Lars', 'LarsCV', 'Lasso', 'LassoCV', 'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LinearRegression', 'LinearSVR', 'MLPRegressor', 'NuSVR', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuitCV', 'PassiveAggressiveRegressor', 'PoissonRegressor', 'RANSACRegressor', 'RandomForestRegressor', 'Ridge', 'RidgeCV', 'SGDRegressor', 'SVR', 'StackingRegressor', 'TheilSenRegressor', 'TransformedTargetRegressor']


REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] in chosen_regressors))
]

reg = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None, regressors=REGRESSORS)
fit,predictions = reg.fit(r.x_train, r.x_test,  r.y_train, r.y_test)

predictions

#--------------------------------------------------------------------
from tpot import TPOTRegressor

# Create an instance of TPOTRegressor
tpot = TPOTRegressor(generations=5, population_size=100, verbosity=2, n_jobs=-1)

# Fit the TPOT model on the training data
tpot.fit(r.x_train, r.y_train)

#results = pd.DataFrame(np.vstack((y_test.values, test_predictions)).T, columns=['True', 'Predicted'])
#print(results)

# Evaluate TPOT regressor on test data
test_predictions = tpot.predict(r.x_test)

print('R2 score:', r2_score(r.y_test, test_predictions))
print('Mean squared error:', mean_squared_error(r.y_test, test_predictions))
print('Mean absolute error:', mean_absolute_error(r.y_test, test_predictions))

# Get the best pipeline
# best_pipeline = tpot.fitted_pipeline_
# Evaluate the best pipeline on the test data
# predictions_final = best_pipeline.predict(r.x_test)
print('R2 score:', r2_score(r.y_test, predictions_final))
print('Mean squared error:', mean_squared_error(r.y_test, predictions_final))
print('Mean absolute error:', mean_absolute_error(r.y_test, predictions_final))

#------------------------------------------------------------
# Get the best pipeline
from sklearn.linear_model import LinearRegression
# Create a linear regression model and fit it to the data
regressor = LinearRegression()
regressor.fit(r.x_train, r.y_train)

# Predict the output for a new input
y_pred = regressor.predict(r.x_test)

print('R2 score:', r2_score(r.y_test, y_pred))
print('Mean squared error:', mean_squared_error(r.y_test, y_pred))
print('Mean absolute error:', mean_absolute_error(r.y_test, y_pred))

