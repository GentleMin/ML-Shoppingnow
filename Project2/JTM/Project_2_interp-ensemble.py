# -*- coding: utf-8 -*-
"""
Project_2.py

Project 2: medical dataset predictions

Introduction to Machine Learning @ ETH Zurich, FS 2022
Group SHOPPINGNOW, author JTM
"""

import os
from pyexpat import model
import numpy as np
import pandas as pd

from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge, RidgeCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

import utils

TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 
         'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

save_suffix = "_dsmm-gb-gb"
verbose = True


"""0. Data reforming"""

# Preliminary: flattening and linear interpolation (if applicable)
if not os.path.exists("./train_features_filled.zip"):
    if verbose:
        print("--- Interpolating training set")
    feat_train = pd.read_csv("./train_features.csv")
    feat_train = utils.patient_feat_flatten(feat_train)
    utils.fill_interp(feat_train, inplace=True)
    feat_train.to_csv("./train_features_filled.zip", header=False, compression="zip")

if not os.path.exists("./test_features_filled.zip"):
    if verbose:
        print("--- Interpolating test set")
    feat_test = pd.read_csv("./test_features.csv")
    feat_test = utils.patient_feat_flatten(feat_test)
    utils.fill_interp(feat_test, inplace=True)
    feat_test.to_csv("./test_features_filled.zip", header=False, compression="zip")

# Load data
feat_train = pd.read_csv("./train_features_filled.zip", header=None, index_col=0, names=utils.flattened_columns()).sort_index()
feat_test = pd.read_csv("./test_features_filled.zip", header=None, index_col=0, names=utils.flattened_columns()).sort_index()
labl_train = pd.read_csv("./train_labels.csv")
labl_train.sort_values("pid", inplace=True)

# Fill NaN entries
feat_train.fillna(0.0, inplace=True)
feat_test.fillna(0.0, inplace=True)

# Manually extract nonlinear features: min + max
feat_train_nonlin = utils.extract_patient_feature(feat_train, fun_list=[lambda x: x.min(axis=1), lambda x: x.max(axis=1)])
feat_train = pd.concat([feat_train, feat_train_nonlin], axis=1).sort_index(axis=1)
feat_test_nonlin = utils.extract_patient_feature(feat_test, fun_list=[lambda x: x.min(axis=1), lambda x: x.max(axis=1)])
feat_test = pd.concat([feat_test, feat_test_nonlin], axis=1).sort_index(axis=1)

# Initialize prediction dataframes
labl_pred_train = pd.DataFrame(index=feat_train.index, columns=labl_train.columns)
labl_pred_train.pid = feat_train.index
labl_pred_test = pd.DataFrame(index=feat_test.index, columns=labl_train.columns)
labl_pred_test.pid = feat_test.index

# Convert to pure matrices
feat_train = feat_train.to_numpy()
feat_test = feat_test.to_numpy()

if verbose:
    print("--- Dataset initialization complete.\n"
          "---     NaN entries filled by interpolation and zero-filling,\n"
          "---     Patient info flattened to vector\n")


"""1. Subtask - test prediction (binary classification)"""

if verbose:
    print("--- SUBTASK-1 MEDICAL TEST PREDICTION ...")

for test_labl in TESTS:
    
    if verbose:
        print("Training:  {} ...".format(test_labl))
    
    # Train with complete dataset, using ensemble Hist Gradient Boosting Randomized Tree
    test_classifier = HistGradientBoostingClassifier(loss="binary_crossentropy", l2_regularization=1.0, random_state=1)
    # test_classifier = GradientBoostingClassifier(loss="deviance", random_state=1)
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
    print("Training:  {} ...".format(sepsis_labl))

# Train with complete dataset, using Logistic Regression
sepsis_classifier = HistGradientBoostingClassifier(loss="binary_crossentropy", l2_regularization=1.0, random_state=1)
# sepsis_classifier = GradientBoostingClassifier(loss="deviance", random_state=1)
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

alphas = np.logspace(-2, 2, 10)
for vital_labl in VITALS:
    if verbose:
        print("Training:  {} ...".format(vital_labl))
            
    # Train with complete dataset, using ridge regression
    # vital_predictor = RidgeCV(alphas=np.logspace(-2, 2, 10), fit_intercept=True, cv=10)
    # vital_predictor.fit(feat_train, labl_train[vital_labl])
    
    # kf = KFold(n_splits=10, shuffle=True, random_state=0)
    # model_est = np.zeros((10, feat_train.shape[1] + 1))
    # for i_split, (i_train, i_test) in enumerate(kf.split(feat_train)):
    #     rms_temp = np.zeros(alphas.size)
    #     for i_reg, alpha in enumerate(alphas):
    #         vital_predictor = Ridge(alpha=alpha, fit_intercept=True).fit(feat_train[i_train, :], labl_train[vital_labl].iloc[i_train])
    #         rms_temp[i_reg] = metrics.mean_squared_error(vital_predictor.predict(feat_train[i_test, :]), labl_train[vital_labl].iloc[i_test])
    #     i_alpha = np.argmin(rms_temp)
    #     vital_predictor = Ridge(alpha=alphas[i_alpha], fit_intercept=True).fit(feat_train[i_train, :], labl_train[vital_labl].iloc[i_train])
    #     model_est[i_split, :-1] = vital_predictor.coef_
    #     model_est[i_split, -1] = vital_predictor.intercept_
    # model_ave = np.mean(model_est, axis=0)
    
    # Train with gradient boosting regressor
    vital_predictor = HistGradientBoostingRegressor(l2_regularization=1.0, random_state=0)
    vital_predictor.fit(feat_train, labl_train[vital_labl])
    
    if verbose:
        print("Recalling: {} ...".format(vital_labl))
            
    # Recall on training set + test set
    labl_pred_train[vital_labl] = vital_predictor.predict(feat_train)
    labl_pred_test[vital_labl] = vital_predictor.predict(feat_test)
    # labl_pred_train[vital_labl] = feat_train @ model_ave[:-1] + model_ave[-1]
    # labl_pred_test[vital_labl] = feat_test @ model_ave[:-1] + model_ave[-1]
    
    if verbose:
        r2 = metrics.r2_score(labl_train[vital_labl], labl_pred_train[vital_labl])
        r2 = 0.5 + 0.5*np.maximum(0, r2)
        print("Recall complete. Normalized R2 = {:.3f}".format(r2))


"""4. Save output"""
labl_pred_train.to_csv(f"./recall_train{save_suffix}.zip", float_format='%.3f', 
                       index=False, compression="zip")
labl_pred_test.to_csv(f"./recall_test{save_suffix}.zip", float_format='%.3f', 
                      index=False, compression="zip")
# labl_pred_train.to_csv(f"./recall_train{save_suffix}.csv", float_format='%.3f', index=False)
# labl_pred_test.to_csv(f"./recall_test{save_suffix}.csv", float_format='%.3f', index=False)
print("Performance on training set:")
print(utils.get_score("./train_labels.csv", f"./recall_train{save_suffix}.zip"))
