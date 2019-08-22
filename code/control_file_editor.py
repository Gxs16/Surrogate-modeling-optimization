#!/usr/bin/env python
# coding: utf-8

# # Control_File_editor

# ## Function-1: read_file()
# Open and read control file

# In[1]:


import re


# In[23]:


def read_file(file_directory):
    control_file = open(file_directory, 'r')
    text = control_file.readlines()
    control_file.close()
    return text


# ## Function-2: modify_file()
# Edit the value in the calib module

# In[21]:


def edit_file(text, key_word, new_value):
    key = re.compile(key_word+'=')
    text[text.index(list(filter(key.search, text))[0])]= key_word+'='+str(new_value)[0:10]+'\n'
    return text


# ## Function-3: save_file()

# In[38]:


def save_file(text, target_directory = ''):
    new_file = open(target_directory+'control_cali.txt','w')
    for line in text:
        new_file.write(line)
    new_file.close()

