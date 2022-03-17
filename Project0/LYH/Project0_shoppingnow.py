#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import csv


# In[2]:


data_train = pd.read_csv('train.csv')


# In[3]:


y_train = data_train.y


# In[4]:


x_train = data_train.iloc[:,2:]


# In[5]:


regr = linear_model.LinearRegression()


# In[6]:


regr.fit(x_train, y_train)


# In[7]:


data_test = pd.read_csv('test.csv')


# In[8]:


y_pred = regr.predict(data_test.iloc[:,1:])


# In[9]:


print("Coefficients: \n", regr.coef_)


# In[10]:


results = np.vstack([data_test.iloc[:,0],y_pred]).T


# In[11]:


header = ['Id', 'y']

with open('test_results_shoppingnow.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    for row in results:
        writer.writerow(row)

