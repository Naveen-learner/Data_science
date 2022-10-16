# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:39:53 2022

@author: Naveen Kumar
"""
#import libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import pylab

#import data file
df = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\calories_consumed.csv")


df.info()

# Checking Whether data is normally distributed
stats.probplot(df['Weight gained (grams)'], dist="norm", plot=pylab) # Quantile-Quantile plot is not perfect

stats.probplot(df['Calories Consumed'], dist="norm", plot=pylab)

#As normal probability plot near to stats plot i am using logarthmic function
# Transformation to make workex variable normal
stats.probplot(np.log(df['Weight gained (grams)']), dist="norm", plot=pylab)
