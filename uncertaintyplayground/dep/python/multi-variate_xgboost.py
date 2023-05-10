"""
A demo for multi-output regression
==================================
The demo is adopted from scikit-learn:
https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_regression_multioutput.html#sphx-glr-auto-examples-ensemble-plot-random-forest-regression-multioutput-py
See :doc:`/tutorials/multioutput` for more information.
"""
# import numpy as np
import xgboost as xgb
# import argparse
# from matplotlib import pyplot as plt
# 
# rng = np.random.RandomState(1994)
# X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
# y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
# y[::5, :] += (0.5 - rng.rand(20, 2))
# y = y - y.min()
# y = y / y.max()

# Train a regressor on it
import xgboost as xgb
reg = xgb.XGBRegressor(tree_method="hist", n_estimators=64)
reg.fit(r.x_comp_tab, r.mod_1[:,0:3], eval_set=[(r.x_comp_tab, r.mod_1[:,0:3])], sample_weight = r.w_comp)
y_predt = reg.predict(r.x_comp_tab)

# need to tune (and fix this tree)

# make a function that can handle this
# can you turn it into a R model
