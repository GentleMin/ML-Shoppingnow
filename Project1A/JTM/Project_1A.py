# -*- coding: utf-8 -*-
"""
Project 1A: Ridge Regression + Multi-fold Cross-Validation

Introduction to Machine Learning @ ETH Zurich, FS 2022
Group SHOPPINGNOW, based on version by Danyang
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df_train = pd.read_csv("./train.csv")
X = df_train.iloc[:, 1:]
y = df_train['y']

# Set up regularization testing and number of split folds
lambda_list = [0.1, 1, 10, 100, 200]
n_split = 10
n_test = 100
rmse_mat = np.ones((len(lambda_list), n_test))

# For each test round
for i in range(n_test):
    
    # Build k-fold splitter
    kf = KFold(n_splits=n_split, shuffle=True)
    
    # calculate RMSE for each fold, each regularization
    rmse_temp = np.ones((len(lambda_list), n_split))
    for i_reg, alpha in enumerate(lambda_list):
        
        for i_split, (train_idx, test_idx) in enumerate(kf.split(X)):
            predictor = Ridge(alpha=alpha, fit_intercept=False).fit(X.iloc[train_idx, :], y.iloc[train_idx])
            rmse_temp[i_reg, i_split] = mean_squared_error(y.iloc[test_idx], predictor.predict(X.iloc[test_idx, :]), squared=False)
    
    # Averaging
    rmse_mat[:, i] = np.mean(rmse_temp, axis=1)
    # print(rmse_mat[:, i])

# Averaging
rmse_reg = np.mean(rmse_mat, axis=1)
print(rmse_reg)

# Result output
pd.DataFrame(data=rmse_reg).to_csv("result.csv", index=False, header=False)

