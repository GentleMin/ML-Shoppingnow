# -*- coding: utf-8 -*-
"""
Project 1A: Ridge Regression + Multi-fold Cross-Validation

Introduction to Machine Learning @ ETH Zurich, FS 2022
Group SHOPPINGNOW, based on version by Danyang
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

"""1. Load Data and set up parameters"""
df_train = pd.read_csv("./train.csv")
X = df_train.iloc[:, 1:]
y = df_train['y']

# Set up regularization testing and number of split folds
lambda_list = [0.1, 1, 10, 100, 200]
n_split = 10
n_test = 10
rmse_mat = np.ones((len(lambda_list), n_test))

"""We note that the RMSE on each validation set is very dispersed.
We take 3 strageties to tackle this randomness:
1. Use k-fold cross-validation, and average over the folds
2. Repeat random splitting several times, and average
3. Use outlier-robust Huber regression
"""

# For each test round
for i in range(n_test):
    
    # Build k-fold splitter
    kf = KFold(n_splits=n_split, shuffle=True)
    
    # calculate RMSE for each fold, each regularization
    rmse_temp = np.ones((len(lambda_list), n_split))

    for i_split, (train_idx, test_idx) in enumerate(kf.split(X)):
        for i_reg, alpha in enumerate(lambda_list):
            predictor = Ridge(alpha=alpha, fit_intercept=False).fit(X.iloc[train_idx, :], y.iloc[train_idx])
            rmse_temp[i_reg, i_split] = mean_squared_error(y.iloc[test_idx], predictor.predict(X.iloc[test_idx, :]), squared=False)
    
    # for i_reg, alpha in enumerate(lambda_list):        
    #     for i_split, (train_idx, test_idx) in enumerate(kf.split(X)):
    #         predictor = Ridge(alpha=alpha, fit_intercept=False).fit(X.iloc[train_idx, :], y.iloc[train_idx])
    #         rmse_temp[i_reg, i_split] = mean_squared_error(y.iloc[test_idx], predictor.predict(X.iloc[test_idx, :]), squared=False)
    
    # Averaging
    rmse_mat[:, i] = np.mean(rmse_temp, axis=1)
    # print(rmse_mat[:, i])


"""A posteriori results show that
Huber regressor is only effective when n_test is relatively small (~10, 30)
At high n_test number, Huber regressor might classify useful points as outliers
and worsen the result
"""

# Averaging
rmse_reg = np.mean(rmse_mat, axis=1)
print(rmse_reg)

# Use robust regressor to estimate the "mean"
# rmse_reg = np.zeros(len(lambda_list))
# for i_reg in range(len(lambda_list)):
#     mean_predictor = HuberRegressor(alpha=0., fit_intercept=False).fit(np.atleast_2d(np.ones(n_test)).T, rmse_mat[i_reg, :])
#     rmse_reg[i_reg] = mean_predictor.coef_
# print(rmse_reg)

# Optional: plot distribution as histograms
# fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
# for i, ax in enumerate(axes):
#     sns.histplot(data=rmse_mat[i, :], ax=ax)
# plt.show()

# Result output
pd.DataFrame(data=rmse_reg).to_csv("result.csv", index=False, header=False)

