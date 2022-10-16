# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 10:02:04 2022

@author: Naveen Kumar
"""

#import necessary libraries
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.cluster import	KMeans

#Read excel using pandas
airlines1 = pd.read_excel(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering\EastWestAirlines.xlsx",'data')
airlines = airlines1.drop(['ID#','cc1_miles','cc2_miles','cc3_miles'], axis = 1)
# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
airlines_norm = norm_func(airlines.iloc[:, :7])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(airlines_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines['clust'] = mb # creating a  new column and assigning it to new column

airlines.head()
airlines_norm.head()

airlines = airlines.iloc[:,[8,7,0,1,2,3,4,5,6]]
airlines.head()

airlines.iloc[:, 2:8].groupby(airlines.clust).mean()

airlines.to_csv("Kmeans_airlines.csv", encoding = "utf-8")

import os
os.chdir(r'C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day13-K-Means Clustering')
