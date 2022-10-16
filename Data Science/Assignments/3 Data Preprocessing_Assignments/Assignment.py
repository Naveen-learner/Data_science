# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:32:05 2022

@author: Naveen Kumar
"""
import numpy as np
import pandas as pd
from scipy import stats
a = [3.9 ,3.9 , 3.85 ,3.08 ,3.15 ,2.76 ,3.21,3.69,3.92,3.92,3.92,3.07,3.07,3.07,2.93,3,3.23,4.08,4.93,4.22,3.7,2.76,3.15,3.73,3.08]
np.mean(a)
np.median(a)
stats.mode(a)
np.var(a)
b = [2.62, 2.875,2.32,3.215,3.44,3.46,3.57,3.19,3.15,3.44,3.44,4.07,3.73,3.78,5.25,5.242,5.345,2.2,1.615,1.835,2.465,3.52,3.435,3.84,3.845]
np.mean(b)
np.median(b)
stats.mode(b)
np.std(b)
np.var(b)
c = [16.46,17.02,18.61,19.44,17.02,20.22,15.84,20,22.9,18.3,18.9,17.4,17.6,18,17.98,17.82,17.42,19.47,18.52,19.9,20.01,16.87,17.3,15.41,17.05]
np.mean(c)
np.median(c)
stats.mode(c)
np.std(c)
np.var(c)

d = [24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00]
d = pd.DataFrame(d)
d.columns=['Data']
import seaborn as sns
sns.boxplot(d['Data'])
d.Data.mean()
d.Data.std()
d.Data.var()
