# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:44:01 2022

@author: Naveen Kumar
"""

#install pandas package
import pandas as pd
#Read data into python
df = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
Education = pd.DataFrame(df, columns=['Data'])
Education
Education.info()
# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
Education.Data.mean() # '.' is used to refer to the variables within object
Education.Data.median()
Education.Data.mode()
# Measures of Dispersion / Second moment business decision
Education.Data.var() # variance
Education.Data.std() # standard deviation
range = max(Education.Data) - min(Education.Data) # range
range
Education.info()
# Third moment business decision
#Skewness=0--symmetrically skewed
#skewness>0--left skewed
#skewness<0--right skewed
Education.Data.skew()
# speed and distance are 
# Fourth moment business decision
#kurtosis>3--leptokurtic
#kurtosis<3--platykurtic
#kurtosis=3--mesokurtic
Education.Data.kurt()


# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

Education.shape

# barplot
plt.bar(height = Education.Data, x = np.arange(1, 19, 1)) # initializing the parameter

plt.hist(Education.Data) # histogram, right -skewed

help(plt.hist)

plt.figure()

plt.boxplot(Education.Data) # boxplot

help(plt.boxplot)