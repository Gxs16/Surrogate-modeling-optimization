#!/usr/bin/env python
# coding: utf-8

# # Training the surrogate model

# In[3]:


from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import sys


# In[4]:


def select_hyperparameters():
    samples = np.loadtxt('original_samples_training.csv', delimiter=',')
    scaler = MinMaxScaler()
    x = samples[:, :-1]
    a = samples[:, -1]
    b = a.reshape(-1,1)
    scaler.fit(b)
    b = scaler.transform(b)
    y = b.reshape(-1,)
    parameters = [
        {'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], \
         'kernel':['linear']},
        {'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], \
         'kernel':['rbf'], \
         'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        {'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], \
         'kernel':['poly'], \
         'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        {'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], \
         'kernel':['sigmoid'], \
         'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    ]
    gs = GridSearchCV(svm.SVR(), parameters, refit=True, cv=10, verbose=2, n_jobs=-1)
    gs.fit(x, y)
    kernel_best = gs.best_params_['kernel']
    C_best = gs.best_params_['C']
    if kernel_best != 'linear':
        gamma_best = gs.best_params_['gamma']

    if kernel_best != 'linear':
        regression = svm.SVR(C = C_best, gamma = gamma_best, kernel = kernel_best,                          cache_size = 1000, max_iter = 5000)
    else:
        regression = svm.SVR(C = C_best, kernel = kernel_best, cache_size = 1000, max_iter = 5000)
    regression.fit(x,y)
    return regression, scaler


# In[ ]:


def train_model(model,scaler):
    samples = np.loadtxt('original_samples_training.csv',delimiter=',')
    x = samples[:, :-1]
    a = samples[:, -1]
    b = a.reshape(-1,1)
    scaler.fit(b)
    b = scaler.transform(b)
    y = b.reshape(-1,)
    return model

