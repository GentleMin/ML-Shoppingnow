
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor


"""I. Linear model, with intercept"""

# 1. Training

# Load data
pd_train = pd.read_csv("./train.csv")
X = pd_train.iloc[:, 2:]
y = pd_train.y

# Train data
predictor = LinearRegression(fit_intercept=True).fit(X, y)

# Training diagnostics
y_est = predictor.predict(X)
rms_misfit = np.sqrt(np.mean(np.abs(y - y_est)**2))
wt = predictor.coef_
intercept = predictor.intercept_

print("Linear model, with intercept ----------------------")
print(f"RMS-misfit = {rms_misfit}")
print(f"Intercept = {intercept}; Linear coefficients: {wt}")

# 2. Testing & Predicting
pd_test = pd.read_csv("./test.csv")
y_test = predictor.predict(pd_test.iloc[:, 1:])

pd_output = pd.DataFrame(data={"Id": pd_test.Id, "y": y_test})
pd_output.to_csv("./result_linear_intercept.csv", index=False)


"""II. Linear model, without intercept"""

# 1. Training

# Load data
pd_train = pd.read_csv("./train.csv")
X = pd_train.iloc[:, 2:]
y = pd_train.y

# Train data
predictor = LinearRegression(fit_intercept=False).fit(X, y)

# Training diagnostics
y_est = predictor.predict(X)
rms_misfit = np.sqrt(np.mean(np.abs(y - y_est)**2))
wt = predictor.coef_

print("Linear model, without intercept -------------------")
print(f"RMS-misfit = {rms_misfit}")
print(f"Intercept = {intercept}; Linear coefficients: {wt}")

# 2. Testing & Predicting
pd_test = pd.read_csv("./test.csv")
y_test = predictor.predict(pd_test.iloc[:, 1:])

pd_output = pd.DataFrame(data={"Id": pd_test.Id, "y": y_test})
pd_output.to_csv("./result_linear.csv", index=False)


"""II. Simply taking the main value"""

# 2. Testing & Predicting
pd_test = pd.read_csv("./test.csv")
y_test = np.mean(pd_test.iloc[:, 1:], axis=1)

pd_output = pd.DataFrame(data={"Id": pd_test.Id, "y": y_test})
pd_output.to_csv("./result_mean.csv", index=False)

