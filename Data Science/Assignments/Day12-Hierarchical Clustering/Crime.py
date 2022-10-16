# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:00:04 2022

@author: Naveen Kumar
"""

#import necessary libraries
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

#Read excel using pandas
crime1 = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering\crime_data.csv")
crime1.rename(columns = {'Unnamed: 0' : 'City'}, inplace = True)

crime1.describe()
crime1.info()


##############################################
### Identify duplicate records in the data ###

duplicate = crime1.duplicated()  # Return boolean Series denoting duplicate rows.
duplicate

sum(duplicate) #No duplicates found

sns.boxplot(crime1['Murder'])
sns.boxplot(crime1['Rape'])
################################################
############## Outlier Treatment ###############

#Winsorization to rectify outliers
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Rape'])
crime1_t = winsor.fit_transform(crime1[['Rape']])
crime1['Rape'] = crime1_t['Rape']


#zero variance
crime1.var()
crime1.var()==0

#################### Missing Values - Imputation ###########################
# Check for count of NA's in each column
crime1.isna().sum()

### Normalization
### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

#drop a column state because it is categorical

crime_norm = norm_func(crime1.iloc[:,1:])
b = crime_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram
# import scipy.cluster.hierarchy as sch

z = linkage(crime_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete', affinity = "euclidean").fit(crime_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

crime1['clust'] = cluster_labels # creating a new column and assigning it to new column 

crime_1 = crime1.iloc[:, [5,0,1,2,3,4]] #re-order columns
crime_1.head()

# Aggregate mean of each cluster for labellling the cluster
#only clustering is success when we are able label it
crime_1.iloc[:, 1:].groupby(crime_1.clust).mean() #groupping by mean clustering
crime_1.iloc[:, 2:].groupby(crime_1.clust).std() #groupping by std clustering
# creating a csv file 
crime_1.to_csv("crimerate.csv", encoding = "utf-8")

import os
os.getcwd() #get current working directory
os.chdir(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering")
