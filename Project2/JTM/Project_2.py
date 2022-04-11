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

# Fill NaN entries
feat_train.fillna(0.0, inplace=True)
feat_test.fillna(0.0, inplace=True)

# Flatten to patient feature vector and sort by pid
feat_train = utils.patient_feat_flatten(feat_train).sort_index()
feat_test = utils.patient_feat_flatten(feat_test).sort_index()
labl_train.sort_values("pid", inplace=True)

# Initialize prediction dataframes
labl_pred_train = pd.DataFrame(index=feat_train.index, columns=labl_train.columns)
labl_pred_train.pid = feat_train.index
labl_pred_test = pd.DataFrame(index=feat_test.index, columns=labl_train.columns)
labl_pred_test.pid = feat_test.index

if verbose:
    print("--- Dataset initialization complete."
          "---     NaN entries filled with 0,"
          "---     Patient info flattened to vector")



"""1. Subtask - test prediction (binary classification)"""

if verbose:
    print("--- SUBTASK-1 MEDICAL TEST PREDICTION ...")

for test_labl in TESTS:
    
    if verbose:
        print("Training:  {} ...".format(test_labl))
    
    # Train with complete dataset, using Logistic Regression
    test_classifier = LogisticRegression(penalty="l2", C=1, fit_intercept=True, 
                                          solver="sag", max_iter=100)
    test_classifier.fit(feat_train, labl_train[test_labl])
    
    if verbose:
        print("Recalling: {} ...".format(test_labl))
    
    # Recall on training set + test set
    labl_pred_train[test_labl] = test_classifier.predict_proba(feat_train)[:, 1]
    labl_pred_test[test_labl] = test_classifier.predict_proba(feat_test)[:, 1]
    
    if verbose:
        roc_auc = metrics.roc_auc_score(labl_train[test_labl], labl_pred_train[test_labl])
        print("Recall complete. ROC-AUC = {:.3f}".format(roc_auc))



"""2. Subtask - predict life-threatening event (septicemia)"""

sepsis_labl = "LABEL_Sepsis"

if verbose:
    print("--- SUBTASK-2 SEPSIS PREDICTION ...")

if verbose:
    print("Training:  {} ...".format(sepsis_labl))

# Train with complete dataset, using Logistic Regression
sepsis_classifier = LogisticRegression(penalty="l2", C=1, fit_intercept=True,
                                       solver="sag", max_iter=100)
sepsis_classifier.fit(feat_train, labl_train[sepsis_labl])

if verbose:
    print("Recalling: {} ...".format(sepsis_labl))

# Recall on training set + test set
labl_pred_train[sepsis_labl] = sepsis_classifier.predict_proba(feat_train)[:, 1]
labl_pred_test[sepsis_labl] = sepsis_classifier.predict_proba(feat_test)[:, 1]

if verbose:
    roc_auc = metrics.roc_auc_score(labl_train[sepsis_labl], labl_pred_train[sepsis_labl])
    print("Recall complete. ROC-AUC = {:.3f}".format(roc_auc))

    

"""3. Subtask - predict vital values"""

if verbose:
    print("--- SUBTASK-3 VITAL PREDICTION ...")

for vital_labl in VITALS:
    
    if verbose:
        print("Training:  {} ...".format(vital_labl))
    
    # Train with complete dataset, using ridge regression
    vital_predictor = Ridge(alpha=1., fit_intercept=True, max_iter=100)
    vital_predictor.fit(feat_train, labl_train[vital_labl])
    
    if verbose:
        print("Recalling: {} ...".format(vital_labl))
    
    # Recall on training set + test set
    labl_pred_train[vital_labl] = vital_predictor.predict(feat_train)
    labl_pred_test[vital_labl] = vital_predictor.predict(feat_test)
    
    if verbose:
        r2 = metrics.r2_score(labl_train[vital_labl], labl_pred_train[vital_labl])
        r2 = 0.5 + 0.5*np.maximum(0, r2)
        print("Recall complete. Normalized R2 = {:.3f}".format(r2))



"""4. Save output"""
labl_pred_train.to_csv(f"./recall_train{save_suffix}.zip", float_format='%.3f', 
                       index=False, compression="zip")
labl_pred_test.to_csv(f"./recall_test{save_suffix}.zip", float_format='%.3f', 
                      index=False, compression="zip")
labl_pred_train.to_csv(f"./recall_train{save_suffix}.csv", float_format='%.3f', index=False)
labl_pred_test.to_csv(f"./recall_test{save_suffix}.csv", float_format='%.3f', index=False)


