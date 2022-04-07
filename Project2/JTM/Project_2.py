import numpy as np
import pandas as pd


def patient_feat_flatten(feat_df: pd.DataFrame):
    
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
    ages_pid = feat_df.loc[:, ["pid", "Age"]].drop_duplicates("pid", keep="first").set_index("pid")
    ages_pid = pd.DataFrame(data=ages_pid, columns=ages_idx)
    
    feat_df_pid = pd.concat([feat_df_pid, ages_pid], axis=1)
    return feat_df_pid
