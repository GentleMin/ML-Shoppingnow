
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor


# 1. Training

# Load data
pd_train = pd.read_csv("./train.csv")
X = np.asarray(pd_train.iloc[:, 2:])
y = np.asarray(pd_train.y)

# Train data
predictor = LinearRegression(fit_intercept=True).fit(X, y)

# Diagnostics
y_est = predictor.predict(X)
rms_misfit = np.sqrt(np.mean(np.abs(y - y_est)**2))
wt = predictor.coef_
intercept = predictor.intercept_
print(f"RMS-misfit = {rms_misfit}")
print(f"Intercept = {intercept}; Linear coefficients: {wt}")

# 2. Testing & Predicting
pd_test = pd.read_csv("./test.csv")
y_test = predictor.predict(np.asarray(pd_test.iloc[:, 1:]))

pd_output = pd.DataFrame(data={"Id": pd_test.Id, "y": y_test})
pd_output.to_csv("./output.csv", index=False)
