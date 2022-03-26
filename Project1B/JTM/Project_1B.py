# -*- coding: utf-8 -*-
"""
Project 1B
Introduction to Machine Learning @ ETH Zurich, FS 2022
Group SHOPPINGNOW
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# import seaborn as sns


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

"""For this specific problem, the condition number is very large.
In this sense, classical regressor might be susceptible to instability and ill-conditioning.
We therefore use regularized regression, and test regularization param via cross-validation.
"""

# cross-validate regularization strength
"""A posteriori results show Ridge/Tikhonov performs slightly better than Lasso"""
alphas = np.logspace(-2, 1, num=30)


# Choosing regularization strength
n_test = 10

rms_err = np.zeros((alphas.size, n_test))
for i in range(n_test):
    kf = KFold(n_splits=10, shuffle=True)
    
    rmse_temp = np.zeros((alphas.size, 10))
    for i_split, (train_idx, test_idx) in enumerate(kf.split(Phi)):
        for i_reg, alpha in enumerate(alphas):
            predictor = Lasso(alpha=alpha, fit_intercept=False).fit(Phi[train_idx, :], y[train_idx])
            rmse_temp[i_reg, i_split] = mean_squared_error(y[test_idx], predictor.predict(Phi[test_idx, :]), squared=False)
    
    rms_err[:, i] = np.mean(rmse_temp, axis=1)

rms_err = np.mean(rms_err, axis=1)
# for i_reg in range(len(alphas)):
#     mean_predictor = HuberRegressor(alpha=0.0, fit_intercept=False).fit(np.ones((n_test, 1)), rms_err[i_reg, :])
#     rms_err[i_reg, 0] = mean_predictor.coef_
# rms_err = rms_err[:, 0]
alpha_idx = np.argmin(rms_err)
alpha_chosen = alphas[alpha_idx]

# training diagnostics
plt.figure(figsize=(8, 5))
plt.semilogx(alphas, rms_err, 'ro')
plt.grid(which="both")
plt.show(block=True)

print("RMS-errors:  {}".format(rms_err))
print("Reg choice: {}".format(alpha_chosen))
print("Condition:  {}".format(np.linalg.cond(Phi.T @ Phi + alpha_chosen * np.identity(Phi.shape[1]))))
# print(predictor.coef_)

"""2. Optional: visualize residual, and decide whether to use outlier-robust method
A posteriori results show that using a robust predictor (Huber) the score is lower.
However judging from the histogram the outliers are not very spurious, therefore using a 
robust regression is not fully justified.
"""

# sns.histplot(y - predictor.predict(Phi))
# plt.show()

predictor = Lasso(alpha=alpha_chosen, fit_intercept=False).fit(Phi, y)
rms_err = np.sqrt(np.mean((y - predictor.predict(Phi))**2))

# training diagnostics
print("RMS-error:  {}".format(rms_err))
print(predictor.coef_)

# output
with open("./results_temp.csv", 'w') as fwrite:
    for ci in predictor.coef_:
        print(ci, file=fwrite)

