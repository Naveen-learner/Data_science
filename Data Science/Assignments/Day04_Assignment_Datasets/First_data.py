# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:59:52 2022

@author: Naveen Kumar
"""
#install pandas package
import pandas as pd
#Read data into python
Education = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day04_Assignment_Datasets\Q1_a.csv")
Education.info()
# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
Education.speed.mean() # '.' is used to refer to the variables within object
Education.speed.median()
Education.speed.mode()
# Measures of Dispersion / Second moment business decision
Education.speed.var() # variance
Education.speed.std() # standard deviation
range = max(Education.speed) - min(Education.speed) # range
range
Education.info()
# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
Education.dist.mean() # '.' is used to refer to the variables within object
Education.dist.median()
Education.dist.mode()
# Measures of Dispersion / Second moment business decision
Education.dist.var() # variance
Education.dist.std() # standard deviation
range = max(Education.dist) - min(Education.dist) # range
range
# Third moment business decision
#Skewness=0--symmetrically skewed
#skewness>0--left skewed
#skewness<0--right skewed
Education.speed.skew()
Education.dist.skew()
# speed and distance are 
# Fourth moment business decision
#kurtosis>3--leptokurtic
#kurtosis<3--platykurtic
#kurtosis=3--mesokurtic
Education.speed.kurt()
Education.dist.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

Education.shape

# barplot
plt.bar(height = Education.dist, x = np.arange(1, 51, 1)) # initializing the parameter
plt.bar(height = Education.speed, x = np.arange(1, 51, 1)) # initializing the parameter

plt.hist(Education.dist) # histogram, right -skewed
plt.hist(Education.speed, color='red')

help(plt.hist)

plt.figure()

plt.boxplot(Education.speed) # boxplot
plt.boxplot(Education.dist) # boxplot

help(plt.boxplot)
