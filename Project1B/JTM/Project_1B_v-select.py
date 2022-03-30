# -*- coding: utf-8 -*-
"""
Project 1B
Introduction to Machine Learning @ ETH Zurich, FS 2022
Group SHOPPINGNOW

Averaging the models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


"""Preliminary: feature map"""
def map_to_feature(X, feature_map, max_n):
    Phi = np.stack([feature_map(X, n) for n in range(max_n)], axis=1)
    return Phi

feature_map_list = [
    lambda X, n: X.iloc[:,n],
    lambda X, n: X.iloc[:,n]**2,
    lambda X, n: np.exp(X.iloc[:,n]),
    lambda X, n: np.cos(X.iloc[:,n]),
    lambda X, n: np.ones(X.iloc[:,n].shape)
]
feature_len_list = [5, 5, 5, 5, 1]


"""1. Train Data"""
df_train = pd.read_csv("./train.csv")
X = df_train.iloc[:,2:]
y = df_train.iloc[:,1]
Phi = np.concatenate([map_to_feature(X, feature_map_list[i], feature_len_list[i]) for i in range(len(feature_len_list))], axis=1)
print("Condition number without regularization: {}".format(np.linalg.cond(Phi.T @ Phi)))

alphas = np.logspace(0, 2.5, num=25)
n_split = 10
n_test = 5
alpha_choices = np.zeros(n_split*n_test)
model_est = np.zeros((n_split*n_test, Phi.shape[1]))

for i in range(n_test):
    kf = KFold(n_splits=n_split, shuffle=True)
    for i_split, (train_idx, test_idx) in enumerate(kf.split(Phi)):
        rms_temp = np.zeros(alphas.size)
        for i_reg, alpha in enumerate(alphas):
            predictor = Ridge(alpha=alpha, fit_intercept=False).fit(Phi[train_idx, :], y[train_idx])
            rms_temp[i_reg] = mean_squared_error(predictor.predict(Phi[test_idx, :]), y[test_idx], squared=False)
        
        alpha_idx = np.argmin(rms_temp)
        alpha_choices[i_split] = alphas[alpha_idx]
        print("Reg choice:", alphas[alpha_idx])
        predictor = Ridge(alpha=alpha, fit_intercept=False).fit(Phi[train_idx, :], y[train_idx])
        model_est[i*n_split + i_split, :] = predictor.coef_

model_ave = np.mean(model_est, axis=0)
print(model_ave)

pd.DataFrame(data=model_ave).to_csv("./results_model-mean.csv", header=False, index=False)
