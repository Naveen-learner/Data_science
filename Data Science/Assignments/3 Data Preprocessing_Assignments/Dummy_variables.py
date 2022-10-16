# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 19:05:46 2022

@author: Naveen Kumar
"""
##################################################
################## Dummy Variables ###############
#Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns

#import data file
data = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\animal_category.csv")

data.columns # column names
data.shape # will give u shape of the dataframe

data.dtypes
data.info()

# Create dummy variables
data_new = pd.get_dummies(data)

data_new_1 = pd.get_dummies(data, drop_first = True)
# Created dummies for all categorical columns

##### One Hot Encoding works
data.columns
data = data[['Animals', 'Gender', 'Homly', 'Types']]

#importing onehot encoder to create dummy variables
from sklearn.preprocessing import OneHotEncoder
# Creating instance of One-Hot Encoder
enc = OneHotEncoder() # initializing method
data.iloc[:,:]

enc_data = pd.DataFrame(enc.fit_transform(data.iloc[:, :]).toarray())

#######################
# Label Encoder
'''from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()

# df['desig'] = labelencoder.fit_transform(df['Position'])

# Data Split into Input and Output variables
X = data.iloc[:, :2]
y = data.iloc[:, 3]

X['Animals'] = labelencoder.fit_transform(X['Animals'])
X['Gender'] = labelencoder.fit_transform(X['Gender'])
X['Homly'] = labelencoder.fit_transform(X['Homly'])'''

