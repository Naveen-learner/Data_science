# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:06:41 2022

@author: Naveen Kumar
"""

import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv("C:\\Users\\Naveen Kumar\\Desktop\\Data Science\\Assignments\\DataSets\\OnlineRetail.csv",encoding='unicode_escape')
data.dtypes
data.UnitPrice = data.UnitPrice.astype('int64')
data.Quantity = data.Quantity.astype('float64')
duplicate = data.duplicated()  # Return boolean Series denoting duplicate rows.
duplicate

sum(duplicate)
# Removing Duplicates
data1 = data.drop_duplicates() # Return DataFrame with duplicate rows removed.
data = data.drop('CustomerID', axis =1)
data.UnitPrice.mean()
data.Quantity.mean()
data.UnitPrice.median()
data.Quantity.median()
data.UnitPrice.mode()
data.Quantity.mode()

range = max(data.UnitPrice) - min(data.UnitPrice) # range
range_1 = max(data.Quantity) - min(data.Quantity) # range

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

data.shape

# barplot
plt.bar(height = data.UnitPrice, x = np.arange(1, 541910, 1)) # initializing the parameter
plt.bar(height = data.Quantity, x = np.arange(1, 541910, 1)) # initializing the parameter
plt.hist(data.UnitPrice) # histogram
plt.hist(data.Quantity, color='red')

help(plt.hist)

plt.figure()

sns.boxplot(data.UnitPrice) # boxplot
sns.boxplot(data.Quantity)

help(plt.boxplot)
