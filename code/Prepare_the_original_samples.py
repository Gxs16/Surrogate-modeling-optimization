#!/usr/bin/env python
# coding: utf-8

# # Prepare the original samples
# Random sample 500 points in parameter space. 

# In[1]:


import numpy as np
import control_file_editor as cfe
import os


# In[ ]:


def compute_nsce():
    dtype = [('times', 'S16')] + [('data', np.float32, 4)]
    output = np.loadtxt('output/ts.chhukha.crest.csv',dtype=dtype, delimiter=',',skiprows=1)
    simulation = output['data'][:,0]
    observation = output['data'][:,1]
    mean_observation = observation.mean()
    num = np.sum((simulation - observation)**2.0)
    den = np.sum((observation - mean_observation)**2.0)
    nsce = num/den
    return nsce


# In[2]:


def generate_bound():
    internal = np.array([[5, 250], [0.1, 20], [0.0099999, 0.5], [0.001, 1], [0, 150],                                      [24.999, 25],[0.00001, 3], [0.01, 1], [1, 10], [0, 0.0000001], [0.01, 3]                                     ,[0.01, 1], [0.01, 5]])
    return internal[:,0], internal[:,1]

lower_bound,upper_bound = generate_bound()


# In[3]:


multiplier = np.random.rand(200, 13)
original_samples = multiplier*(upper_bound - lower_bound) + lower_bound


# In[7]:

parameters = ['wm', 'b', 'im', 'ke', 'fc', 'iwu', 'under', 'leaki', 'th', 'isu', 'alpha', 'beta', 'alpha0']
file_name = cfe.read_file('control_cali.txt')
nsce = np.empty([200,1])
for line_samples in range(original_samples.shape[0]):
    for row_samples in range(original_samples.shape[1]):
        text = cfe.edit_file(file_name, parameters[row_samples], original_samples[line_samples, row_samples])
        #print(text)
    cfe.save_file(text)
    os.system('nohup /hydros/MengyuChen/ef5_code/bin/ef5 control_cali.txt')
    #os.system('cat control_cali.txt')
    nsce_sample = compute_nsce()
    nsce[line_samples] = nsce_sample
original_samples = np.hstack((original_samples, nsce))
original_samples_training = np.hstack((multiplier, nsce))
np.savetxt('original_samples.csv',original_samples,delimiter=',')
np.savetxt('original_samples_training.csv',original_samples_training,delimiter=',')
