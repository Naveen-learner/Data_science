# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:11:53 2022

@author: Naveen Kumar
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

data_1 = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\cities.csv")

data = data_1.drop(["CITY"], axis=1)
data.dtypes
duplicate = data.duplicated()  # Return boolean Series denoting duplicate rows.
duplicate

sum(duplicate)

sns.boxplot(data.WORK)

sns.boxplot(data.PRICE)
sns.boxplot(data.SALARY)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['WORK'])
data['WORK'] = winsor.fit_transform(data[['WORK']])

data.var() # variance of numeric variables
data.var() == 0
data.var(axis=0) == 0

data.isna().sum()

data_new = pd.get_dummies(data,drop_first=True)

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

data_norm = norm_func(data_new)
b = data_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram
# import scipy.cluster.hierarchy as sch

z = linkage(data_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete', affinity = "euclidean").fit(data_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data_1['clust'] = cluster_labels # creating a new column and assigning it to new column 



# Aggregate mean of each cluster for labellling the cluster
#only clustering is success when we are able label it
data_1.iloc[:, 2:].groupby(data_1.clust).mean() #groupping by mean clustering
data_1.iloc[:, 2:].groupby(data_1.clust).std() #groupping by std clustering
# creating a csv file 
Univ1.to_csv("University.csv", encoding = "utf-8")

import os
os.getcwd() #get current working directory
os.chdir(r"C:\Users\Naveen Kumar\Desktop\Data Science\Class\Datasets")

