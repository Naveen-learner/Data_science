# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 10:45:01 2022

@author: Naveen Kumar
"""
#import necessary libraries
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.cluster import	KMeans
import os

#Read excel using pandas
crime1 = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering\crime_data.csv")
crime1.rename(columns = {'Unnamed: 0' : 'City'}, inplace = True)
crime1.describe()
crime1.info()
#Winsorization to rectify outliers
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Rape'])
crime1_t = winsor.fit_transform(crime1[['Rape']])
crime1['Rape'] = crime1_t['Rape']

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

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(crime_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime1['clust'] = mb # creating a  new column and assigning it to new column 

crime1.head()
crime_norm.head()

crime1 = crime1.iloc[:,[5,0,1,2,3,4]]
crime1.head()

crime1.iloc[:, 2:6].groupby(crime1.clust).mean()
os.chdir(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering")
crime1.to_csv("Kmeans_crime.csv", encoding = "utf-8")
