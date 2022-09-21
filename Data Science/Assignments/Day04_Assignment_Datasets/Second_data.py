# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:36:04 2022

@author: Naveen Kumar
"""

#install pandas package
import pandas as pd
#Read data into python
Education = pd.read_csv(r"C:\\Users\\Naveen Kumar\\Desktop\\Data Science\\Assignments\\Day04_Assignment_Datasets\\Q2_b.csv")
Education.info()
# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
Education.SP.mean() # '.' is used to refer to the variables within object
Education.SP.median()
Education.SP.mode()
# Measures of Dispersion / Second moment business decision
Education.SP.var() # variance
Education.SP.std() # standard deviation
range = max(Education.SP) - min(Education.SP) # range
range
Education.info()
# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
Education.WT.mean() # '.' is used to refer to the variables within object
Education.WT.median()
Education.WT.mode()
# Measures of Dispersion / Second moment business decision
Education.WT.var() # variance
Education.WT.std() # standard deviation
range = max(Education.WT) - min(Education.WT) # range
range
# Third moment business decision
#Skewness=0--symmetrically skewed
#skewness>0--left skewed
#skewness<0--right skewed
Education.SP.skew()
Education.WT.skew()
# speed and distance are 
# Fourth moment business decision
#kurtosis>3--leptokurtic
#kurtosis<3--platykurtic
#kurtosis=3--mesokurtic
Education.SP.kurt()
Education.WT.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

Education.shape

# barplot
plt.bar(height = Education.SP, x = np.arange(1, 82, 1)) # initializing the parameter
plt.bar(height = Education.WT, x = np.arange(1, 82, 1)) # initializing the parameter

plt.hist(Education.SP) # histogram, right -skewed
plt.hist(Education.WT, color='red')

help(plt.hist)

plt.figure()

plt.boxplot(Education.SP) # boxplot
plt.boxplot(Education.WT) # boxplot

help(plt.boxplot)
