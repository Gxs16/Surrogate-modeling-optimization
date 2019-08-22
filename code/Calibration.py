#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
import sys
import os
import control_file_editor as cfe
import Surrogate_model as Sm
import Prepare_the_original_samples as Pos
import ASMO


# In[ ]:


def generate_new_sample(old_sample_file, new_sample_point):
    old_sample_training = np.loadtxt(old_sample_file, delimiter=',')
    new_sample_training = np.vstack((old_sample_training,new_sample_point))
    np.savetxt(old_sample_file,new_sample_training, delimiter=',')

def generate_bound():
    internal = np.array([[5, 250], [0.1, 20], [0.0099999, 0.5], [0.001, 1], [0, 150],                                      [24.999, 25],[0.00001, 3], [0.01, 1], [1, 10], [0, 0.0000001], [0.01, 3]                                     ,[0.01, 1], [0.01, 5]])
    return internal[:,0], internal[:,1]

def compute_nsce():
    dtype = [('times', 'S16')] + [('data', np.float32, 4)]
    output = np.loadtxt('output/ts.chhukha.crest.csv',dtype=dtype, delimiter=',',skiprows=1)
    simulation = output['data'][:,0]
    observation = output['data'][:,1]
    mean_observation = observation.mean()
    num = np.sum((simulation - observation)**2.0)
    den = np.sum((observation - mean_observation)**2.0)
    nsce = -1.0 + num/den
    return nsce
# In[ ]:


lower_bound,upper_bound = generate_bound()
xlb = np.zeros(13)
xub = np.ones(13)
nInput = 13
niter = 50
parameters = ['wm', 'b', 'im', 'ke', 'fc', 'iwu', 'under', 'leaki', 'th', 'isu', 'alpha', 'beta', 'alpha0']

# In[ ]:
nsce = 1
loop = 1
model, scaler = Sm.select_hyperparameters()
while nsce >= 0.3:
    Sm.train_model(model, scaler)
    bestx, bestf, x, f = ASMO.optimization(model, nInput, xlb, xub, niter)
    
    new_parameter = bestx*(upper_bound - lower_bound) + lower_bound
    file = cfe.read_file('control_cali.txt')
    
    for index_param in range(new_parameter.shape[0]):
        text = cfe.edit_file(file, parameters[index_param], new_parameter[index_param])
    cfe.save_file(text)
    
    os.system('/hydros/MengyuChen/ef5_code/bin/ef5 control_cali.txt')
    
    nsce = compute_nsce()

    new_surrogate_point = np.hstack((bestx,nsce))
    new_simulate_point = np.hstack((new_parameter,nsce))
    #save file
    generate_new_sample('original_samples_training.csv', new_surrogate_point)
    generate_new_sample('original_samples.csv', new_simulate_point)
    print('cali_loop: ',loop)
    print('nsce: ',nsce)
    loop+=1
print(new_parameter)

