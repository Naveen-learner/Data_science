2 + 2 # Function F9
# Works as calculator

# Python Libraries (Packages)
# pip install <package name> - To install library (package), execute the code in Command prompt
# pip install pandas

import pandas as pd

dir(pd)

# Read data into Python
data = pd.read_csv("C:/Users/Naveen Kumar/Desktop/Data Science/Class/Datasets/education.csv")
data.dtype #Education = pd.read_csv("C:/Data/education.csv")

A=10
a=10.1

education.info()

# C:\Users\education.csv - this is windows default file path with a '\'
# C:\\Users\\education.csv - change it to '\\' to make it work in Python

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
education.workex.mean() # '.' is used to refer to the variables within object
education.workex.median()
education.workex.mode()

# Measures of Dispersion / Second moment business decision
education.workex.var() # variance
education.workex.std() # standard deviation
range = max(education.workex) - min(education.workex) # range
range

# Third moment business decision
education.workex.skew()
education.gmat.skew()

# Fourth moment business decision
education.workex.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

education.shape

# barplot
plt.bar(height = education.gmat, x = np.arange(1, 774, 1)) # initializing the parameter

plt.hist(education.gmat) # histogram
plt.hist(education.workex, color='red')

help(plt.hist)

plt.figure()

plt.boxplot(education.gmat) # boxplot

help(plt.boxplot)
