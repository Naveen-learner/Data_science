# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:59:29 2022

@author: Naveen Kumar
"""

import pandas as pd

data = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\Z_dataset.csv")
data.dtypes
duplicate = data.duplicated()
sum(duplicate)

##############################################
#### zero variance and near zero variance ####

# If the variance is low or close to zero, then a feature is approximately 
# constant and will not improve the performance of the model.
# In that case, it should be removed. 

data.var() # variance of numeric variables
data.var() == 0
data.var(axis=0) == 0

#####another method to remove zero variance#######
#import variance threshold from sklearn.featur-selection
from sklearn.feature_selection import VarianceThreshold
#Takiing threshold variance as 90%
sel = VarianceThreshold(threshold=(.9 * (1 - .9)))  
#Transforming the data to threshold variance  
sel.fit_transform(data.iloc[:, 1:5])
data_v=sel.fit_transform(data.iloc[:, 1:5])
