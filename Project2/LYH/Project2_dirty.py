# -*- coding: utf-8 -*-
"""
Project_2.py

Project 2: medical dataset predictions

Introduction to Machine Learning @ ETH Zurich, FS 2022
Group SHOPPINGNOW, author JTM
"""


import numpy as np
import pandas as pd

from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

import numpy.linalg as linalg
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import utils


TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 
         'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

save_suffix = ""
verbose = True


"""0. Data reforming"""

if verbose:
    print("--- PREPARING DATASET ...")

# Load data
feat_train = pd.read_csv("./train_features.csv")
labl_train = pd.read_csv("./train_labels.csv")
feat_test = pd.read_csv("./test_features.csv")

# Drop feature with more than 90% NaN
# feat_train = feat_train.dropna(axis=1,thresh=feat_train.shape[0]/10)
# labl_train = labl_train.dropna(axis=1,thresh=labl_train.shape[0]/10)
# feat_test = feat_test.dropna(axis=1,thresh=feat_test.shape[0]/10)

# Fill NaN entries
feat_train.fillna(0.0, inplace=True)
feat_test.fillna(0.0, inplace=True)

# Fill NaN using IterativeImputer
# imput = IterativeImputer(random_state=0)
# feat_train = pd.DataFrame(imput.fit_transform(feat_train), columns = feat_train.columns)
# feat_test = pd.DataFrame(imput.fit_transform(feat_test), columns = feat_test.columns)


# Fill NaN using "ffill" and "bfill"
# feat_train.fillna(method="bfill", inplace=True)
# feat_train.fillna(method="ffill", inplace=True)
# feat_test.fillna(method="bfill", inplace=True)
# feat_test.fillna(method="ffill", inplace=True)

# Normalize the data (max-min)
# feat_train.iloc[:,3:] = (feat_train.iloc[:,3:]-feat_train.iloc[:,3:].min())/(feat_train.iloc[:,3:].max()-feat_train.iloc[:,3:].min())
# feat_test.iloc[:,3:] = (feat_test.iloc[:,3:]-feat_test.iloc[:,3:].min())/(feat_test.iloc[:,3:].max()-feat_test.iloc[:,3:].min())


# Flatten to patient feature vector and sort by pid
feat_train = utils.patient_feat_flatten(feat_train).sort_index()
feat_test = utils.patient_feat_flatten(feat_test).sort_index()
labl_train.sort_values("pid", inplace=True)

# Initialize prediction dataframes
labl_pred_train = pd.DataFrame(index=feat_train.index, columns=labl_train.columns)
labl_pred_train.pid = feat_train.index
labl_pred_test = pd.DataFrame(index=feat_test.index, columns=labl_train.columns)
labl_pred_test.pid = feat_test.index


"""1. Subtask - test prediction (binary classification)"""

for test_labl in TESTS:
        
    # Train with complete dataset, using Logistic Regression (with stronger regulation and more iter)
    test_classifier = HistGradientBoostingClassifier(loss="binary_crossentropy",l2_regularization=1, early_stopping="auto", validation_fraction=0.1, 
                                          n_iter_no_change=10)
    test_classifier.fit(feat_train, labl_train[test_labl])
        
    # Recall on training set + test set
    labl_pred_train[test_labl] = test_classifier.predict_proba(feat_train)[:, 1]
    labl_pred_test[test_labl] = test_classifier.predict_proba(feat_test)[:, 1]


"""2. Subtask - predict life-threatening event (septicemia)"""

sepsis_labl = "LABEL_Sepsis"

# Train with complete dataset, using Logistic Regression
sepsis_classifier = HistGradientBoostingClassifier(loss="binary_crossentropy",l2_regularization=1, early_stopping="auto", validation_fraction=0.1, 
                                          n_iter_no_change=10)
sepsis_classifier.fit(feat_train, labl_train[sepsis_labl])

# Recall on training set + test set
labl_pred_train[sepsis_labl] = sepsis_classifier.predict_proba(feat_train)[:, 1]
labl_pred_test[sepsis_labl] = sepsis_classifier.predict_proba(feat_test)[:, 1]


"""3. Subtask - predict vital values"""

for vital_labl in VITALS:
        
    # Train with complete dataset, using ridge regression
    vital_predictor = Ridge(alpha=1, fit_intercept=True, max_iter=100)
    vital_predictor.fit(feat_train, labl_train[vital_labl])
        
    # Recall on training set + test set
    labl_pred_train[vital_labl] = vital_predictor.predict(feat_train)
    labl_pred_test[vital_labl] = vital_predictor.predict(feat_test)

"""4. Save output"""
labl_pred_train.to_csv(f"./recall_train{save_suffix}.zip", float_format='%.3f', 
                       index=False, compression="zip")
labl_pred_test.to_csv(f"./recall_test{save_suffix}.zip", float_format='%.3f', 
                      index=False, compression="zip")
labl_pred_train.to_csv(f"./recall_train{save_suffix}.csv", float_format='%.3f', index=False)
labl_pred_test.to_csv(f"./recall_test{save_suffix}.csv", float_format='%.3f', index=False)