# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 12:00:53 2022

@author: Naveen Kumar
"""

#import necessary libraries
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
import numpy as np
from sklearn.cluster import	KMeans

#Read excel using pandas
Insurance = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering\Autoinsurance.csv")

Insurance1 = Insurance.drop(['Customer','State'],axis =1 )
Insurance1.describe()
Insurance1.info()

duplicate = Insurance1.duplicated()  # Return boolean Series denoting duplicate rows.
sum(duplicate) #There are no duplicates found
Insurance1.drop_duplicates(inplace=True) # Return DataFrame with duplicate rows removed.

#To check outliers
sns.boxplot(Insurance1['Customer Lifetime Value'])#outliers present
sns.boxplot(Insurance1['Income'])
sns.boxplot(Insurance1['Monthly Premium Auto'])#outliers present
sns.boxplot(Insurance1['Months Since Last Claim'])
sns.boxplot(Insurance1['Months Since Policy Inception'])
sns.boxplot(Insurance1['Number of Open Complaints'])#outliers present
sns.boxplot(Insurance1['Number of Policies'])#outliers present
sns.boxplot(Insurance1['Total Claim Amount'])#outliers present

#Winsorization to rectify outliers
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Customer Lifetime Value','Monthly Premium Auto','Number of Open Complaints','Number of Policies','Total Claim Amount'])

Insurance1_t = winsor.fit_transform(Insurance1[['Customer Lifetime Value','Monthly Premium Auto','Number of Open Complaints','Number of Policies','Total Claim Amount']])
Insurance1['Customer Lifetime Value'] = Insurance1_t['Customer Lifetime Value']
Insurance1['Monthly Premium Auto'] = Insurance1_t['Monthly Premium Auto']
Insurance1['Number of Open Complaints'] = Insurance1_t['Number of Open Complaints']
Insurance1['Number of Policies'] = Insurance1_t['Number of Policies']
Insurance1['Total Claim Amount'] = Insurance1_t['Total Claim Amount']

Insurance1.describe()
Insurance1.info()
#zero variance
Insurance1.var()
Insurance1.var()==0
Insurance1.drop(['Number of Open Complaints'], axis = 1, inplace =True)

# Label Encoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Insurance1['Coverage'] = labelencoder.fit_transform(Insurance1['Coverage'])
Insurance1['Education'] = labelencoder.fit_transform(Insurance1['Education'])
Insurance1['Location Code'] = labelencoder.fit_transform(Insurance1['Location Code'])
Insurance1['Vehicle Class'] = labelencoder.fit_transform(Insurance1['Vehicle Class'])
Insurance1 = pd.get_dummies(Insurance1, drop_first = True)

### Normalization
### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

Insurance_norm = norm_func(Insurance1)
b = Insurance_norm.describe()
Insurance_norm.isna().sum()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Insurance_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(Insurance_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Insurance1['clust'] = mb # creating a  new column and assigning it to new column 

Insurance1.head()
Insurance_norm.head()

Insurance1 = Insurance1.iloc[:,[95,0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94]]
Insurance1.head()

Insurance3 = Insurance1.iloc[:, 1:].groupby(Insurance1.clust).mean()
os.chdir(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering")
Insurance1.to_csv("Kmeans_Insurance.csv", encoding = "utf-8")
