### code for Task2a iml
### Team: ShoppingNow

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import glob

train_labels = pd.read_csv('./train_labels.csv')
labels = train_labels.iloc[:,1:11]
labels_names = labels.columns

train_features = pd.read_csv('./train_features.csv')
features_names = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate',\
                  'TroponinI', 'SaO2', 'Bilirubin_direct', 'EtCO2']

test_features = pd.read_csv('./test_features.csv')
result = train_labels.drop_duplicates(subset='pid', keep=False).pid

if next(glob.iglob("./*_test.csv"), None) and next(glob.iglob("./*_train.csv"), None):
    print("there are txt files") # there are text files
else:
    print("there are no text files")

    train_fill = train_features
    test_fill = test_features

    # split different data for each patients
    for f_name in features_names:
        print(f_name) # to visualize the procedue

        # replace the NaN with the average
        train_fill[f_name] = train_fill[f_name].replace(np.nan, train_features[f_name].mean())
        # group for different patient
        gt = train_features.groupby('pid',as_index=False)[f_name]
        # group_keys = gt.groups.keys()

        # get resutls of one medical test for all patients
        train_data = pd.DataFrame()
        for i in gt.groups:
            patient = gt.get_group(i).reset_index(drop=True, inplace=False)
            train_data = pd.concat([train_data,patient.rename('patient'+str(i))],axis=1)
        (train_data.T).to_csv(f_name+"_train.csv",index=False,header=True)

        test_fill[f_name] = test_fill[f_name].replace(np.nan, test_features[f_name].mean()) # replace the NaN with the average
        gt_test = test_features.groupby('pid',as_index=False)[f_name] # group for different patient
        test_data = pd.DataFrame()
        for i in gt_test.groups:
            patient_test = gt_test.get_group(i).reset_index(drop=True, inplace=False)
            test_data = pd.concat([test_data,patient_test.rename('patient'+str(i))],axis=1)
        (test_data.T).to_csv(f_name+"_test.csv",index=False, header=True)

# split different data for each patients
for (f_name, l_name) in zip(features_names, labels_names):
    x = pd.read_csv('./'+f_name+'_train.csv')
    x_test = pd.read_csv('./'+f_name+'_test.csv')

    y = labels[l_name]

    clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(x, y)
    y_pre = clf.predict(x_test)
    temp = pd.DataFrame(y_pre, columns = [l_name])
    result = pd.concat([result,temp], axis=1)

result.to_csv("result.csv",index=False,header=False)
