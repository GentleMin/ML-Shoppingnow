import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, HuberRegressor
import matplotlib.pyplot as plt
import seaborn as sns


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

# cross-validate regularization strength
alphas = np.logspace(-1, 2, num=31)
predictor = RidgeCV(alphas=alphas, cv=10, fit_intercept=False).fit(Phi, y)
rms_err = np.sqrt(np.mean((y - predictor.predict(Phi))**2))
alpha_chosen = predictor.alpha_

# training diagnostics
print("RMS-error:  {}".format(rms_err))
print("Reg choice: {}".format(alpha_chosen))
print("Condition:  {}".format(np.linalg.cond(Phi.T @ Phi + alpha_chosen * np.identity(Phi.shape[1]))))
print(predictor.coef_)

"""2. Optional: visualize residual, and decide whether to use outlier-robust method
A posteriori results show that using a robust predictor (Huber) the score is lower.
However judging from the histogram the outliers are not very spurious, therefore using a 
robust regression is not fully justified.
"""
sns.histplot(y - predictor.predict(Phi))
plt.show()

predictor = HuberRegressor(alpha=alpha_chosen, fit_intercept=False).fit(Phi, y)
rms_err = np.sqrt(np.mean((y - predictor.predict(Phi))**2))

# training diagnostics
print("RMS-error:  {}".format(rms_err))
print(predictor.coef_)

# output
with open("./results_Ridge-CV10-Huber.csv", 'w') as fwrite:
    for ci in predictor.coef_:
        print(ci, file=fwrite)

