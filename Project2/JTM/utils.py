# -*- coding: utf-8 -*-
"""
utils.py

Utility collections for task 2

Introduction to Machine Learning @ ETH Zurich, FS 2022
Group SHOPPINGNOW, author JTM
"""

import numpy as np
import pandas as pd
import sklearn.metrics as metrics


VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


def patient_feat_flatten(feat_df: pd.DataFrame):
    """Flatten all patients data into a dataframe, where each line corresponds to 
    the measures of one single patient.
    """
    
    col_idx_levels = [feat_df.columns[3:], list(range(12))]
    col_idx = pd.MultiIndex.from_product(col_idx_levels, names=["Measure", "Time"])
    
    # Flatten time-dependent measurement series to single vector
    pid_list = list(feat_df.drop_duplicates(subset="pid", keep="first").pid)
    feat_gp_pid = feat_df.groupby("pid", sort=False)
    feat_list = [feat_gp.iloc[:, 3:].to_numpy().flatten(order='F') for pid, feat_gp in feat_gp_pid]
    feat_list = np.stack(feat_list, axis=0)
    feat_df_pid = pd.DataFrame(data=feat_list, index=pid_list, columns=col_idx)
    
    # Add time-independent measurement: age
    ages_idx = pd.MultiIndex.from_tuples([("Age", 0)], names=["Measure", "Time"])
    ages_pid = feat_df.loc[:, ["pid", "Age"]].drop_duplicates(subset="pid", keep="first").set_index("pid")
    ages_pid = pd.DataFrame(data=ages_pid.to_numpy(), index=ages_pid.index, columns=ages_idx)
    
    feat_df_pid = pd.concat([feat_df_pid, ages_pid], axis=1)
    return feat_df_pid


def get_score(df_true, df_pred):
    """Standard score calculation based on true labels.
    Input parameters can be either pandas.DataFrame instances or strings of file paths.
    If inputs are strings, the dataframes will be read based on the file path.
    
    :param df_true: dataframe of true labels
    :param df_pred: dataframe of predictions, ideally probabilities between [0, 1]
    """
    
    if isinstance(df_true, str):
        df_true = pd.read_csv(df_true)
    if isinstance(df_pred, str):
        df_pred = pd.read_csv(df_pred)
    
    df_true = df_true.sort_values('pid')
    df_pred = df_pred.sort_values('pid')
    
    task1_scores = [metrics.roc_auc_score(df_true[entry], df_pred[entry]) for entry in TESTS]
    task1 = np.mean(task1_scores)
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_pred['LABEL_Sepsis'])
    task3_scores = [0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_pred[entry])) for entry in VITALS]
    task3 = np.mean(task3_scores)
    score = np.mean([task1, task2, task3])
    
    print(task1_scores)
    print(task3_scores)
    print(task1, task2, task3)
    
    return score


def fill_interp(df_patients: pd.DataFrame, inplace=False):
    """Fill nan values by linear interpolation.
    
    This method should be called on a patient-flattened dataframe, 
    i.e. a dataframe whose each row corresponds to one single patient.
    Such dataframe is produced by `patient_feat_flatten` method.
    """
    if not inplace:
        df_output = df_patients.copy(deep=True)
    else:
        df_output = df_patients
    
    # obtain all measured quantities
    measurements = df_patients.columns.get_level_values(0).drop_duplicates(keep="first")
    
    # Loop through patients
    patient_count = 0
    patient_total = df_patients.shape[0]
    for pid in df_patients.index:
        
        # Loop through measurements
        for measure in measurements:
            
            measured_array = df_patients.loc[pid, measure]
            
            # Scalar measurement - non-time-series: nothing to interpolate
            if measured_array.size == 1:
                continue
            
            # linear interpolate with valid numeric data
            time_array = measured_array.index
            valid_idx = ~measured_array.isna()
            
            # If we have 12 nan measurements, cannot interpolate
            if np.sum(valid_idx) == 0:
                continue
            
            interp_array = np.interp(time_array, time_array[valid_idx], measured_array[valid_idx])
            
            # fill
            df_output.loc[pid, measure] = interp_array
        
        patient_count += 1
        if patient_count%100 == 0:
            print("{}/{}".format(patient_count, patient_total))
    
    return df_output
