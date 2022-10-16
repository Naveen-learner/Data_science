# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:52:43 2022

@author: Naveen Kumar
"""

#import necessary libraries
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

#Read excel using pandas
airlines1 = pd.read_excel(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering\EastWestAirlines.xlsx",'data')

airlines1.describe()
airlines1.info()

#Typecasting(as mentioned cc miles are char)
airlines1['cc1_miles'] = airlines1['cc1_miles'].astype(str)
airlines1['cc2_miles'] = airlines1['cc2_miles'].astype(str)
airlines1['cc3_miles'] = airlines1['cc3_miles'].astype(str)

##############################################
### Identify duplicate records in the data ###

duplicate = airlines1.duplicated()  # Return boolean Series denoting duplicate rows.
duplicate

sum(duplicate) #No duplicates found

################################################
############## Outlier Treatment ###############

#Winsorization to rectify outliers
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Balance','Qual_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12'])
airlines1_t = winsor.fit_transform(airlines1[['Balance','Qual_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12']])

airlines1['Balance'] = airlines1_t['Balance']
airlines1['Qual_miles'] = airlines1_t['Qual_miles']
airlines1['Bonus_miles'] = airlines1_t['Bonus_miles']
airlines1['Bonus_trans'] = airlines1_t['Bonus_trans']
airlines1['Flight_miles_12mo'] = airlines1_t['Flight_miles_12mo']
airlines1['Flight_trans_12'] = airlines1_t['Flight_trans_12']

#zero variance
airlines1.var()
airlines1.var()==0
airlines1.drop(['Qual_miles'], axis = 1, inplace =True) #dropping qualmiles because of zero variance
#################### Missing Values - Imputation ###########################
# Check for count of NA's in each column
airlines1.isna().sum()

airlines = pd.get_dummies(airlines1, drop_first = True)
### Normalization
### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

#drop a column state because it is categorical
airlines = airlines1.drop(["ID#",'cc1_miles','cc2_miles','cc3_miles'], axis=1)
airlines_norm = norm_func(airlines.iloc[:,:6])
b = airlines_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram
# import scipy.cluster.hierarchy as sch

z = linkage(airlines_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 7, linkage = 'complete', affinity = "euclidean").fit(airlines_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

airlines['clust'] = cluster_labels # creating a new column and assigning it to new column 

airlines_1 = airlines.iloc[:, [7,6,0,1,2,3,4,5]] #re-order columns
airlines_1.head()

# Aggregate mean of each cluster for labellling the cluster
#only clustering is success when we are able label it
airlines_1.iloc[:, 1:].groupby(airlines_1.clust).mean() #groupping by mean clustering
airlines_1.iloc[:, 1:].groupby(airlines_1.clust).std() #groupping by std clustering
# creating a csv file 
airlines_1.to_csv("EastWestAirlines.csv", encoding = "utf-8")

import os
os.getcwd() #get current working directory
os.chdir(r"C:\Users\Naveen Kumar\Desktop\Data Science\Class\Datasets")
