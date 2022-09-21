# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:38:13 2022

@author: Naveen Kumar
"""
#importing pandas library
import pandas as pd

#############
# Discretization

data = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\iris.csv")
data.dtypes
data.head()

data.info()

data.describe()
#Renaming columns as they contain '.'
data = data.rename({'Sepal.Length':'Length', 'Sepal.Width':'Width', 'Petal.Length':'Plength', 'Petal.Width':'Pwidth'},axis=1)
#discretization of four numerical columns ['low','high'] as labels
data['Length_new'] = pd.cut(data['Length'], 
                              bins = [min(data.Length),
                                      data.Length.mean(),
                                      max(data.Length)],
                              labels=["Low", "High"])

data['Width_new'] = pd.cut(data['Width'], 
                              bins = [min(data.Width),
                                      data.Width.mean(),
                                      max(data.Width)],
                              labels=["Low", "High"])

data['Plength_new'] = pd.cut(data['Plength'], 
                              bins = [min(data.Plength),
                                      data.Plength.mean(),
                                      max(data.Plength)],
                              labels=["Low", "High"])

data['Pwidth_new'] = pd.cut(data['Pwidth'], 
                              bins = [min(data.Pwidth),
                                      data.Pwidth.mean(),
                                      max(data.Pwidth)],
                              labels=["Low", "High"])

data.head(10)
data.Length_new.value_counts()
data.Width_new.value_counts()
data.Plength_new.value_counts()
data.Pwidth_new.value_counts()
