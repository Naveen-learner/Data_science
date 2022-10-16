# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:20:13 2022

@author: Naveen Kumar
"""

#import necessary libraries
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.cluster import	KMeans
import os

#Read excel using pandas
Telecom = pd.read_excel(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering\Telco_customer_churn.xlsx")

Telecom_1 = Telecom.drop(['Customer ID'],axis =1 )
Telecom_1.describe()
Telecom_1.info()

duplicate = Telecom_1.duplicated()  # Return boolean Series denoting duplicate rows.
sum(duplicate) #There are no duplicates found

#Winsorization to rectify outliers
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Number of Referrals','Avg Monthly GB Download','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue'])

Telecom_1_t = winsor.fit_transform(Telecom_1[['Number of Referrals','Avg Monthly GB Download','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']])
Telecom_1['Number of Referrals'] = Telecom_1_t['Number of Referrals']
Telecom_1['Avg Monthly GB Download'] = Telecom_1_t['Avg Monthly GB Download']
Telecom_1['Total Refunds'] = Telecom_1_t['Total Refunds']
Telecom_1['Total Extra Data Charges'] = Telecom_1_t['Total Extra Data Charges']
Telecom_1['Total Long Distance Charges'] = Telecom_1_t['Total Long Distance Charges']
Telecom_1['Total Revenue'] = Telecom_1_t['Total Revenue']

Telecom_1.describe()
Telecom_1.info()
#zero variance
Telecom_1.var()
Telecom_1.var()==0
Telecom_1.drop(['Count','Total Refunds','Total Extra Data Charges'], axis = 1, inplace =True)

#################### Missing Values - Imputation ###########################
# Check for count of NA's in each column
Telecom_1.isna().sum()
# Create dummy variables

Telecom_new = pd.get_dummies(Telecom_1, drop_first = True)
# Created dummies for all categorical columns

### Normalization
### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

Telecom_norm = norm_func(Telecom_new)
b = Telecom_norm.describe()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Telecom_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(Telecom_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Telecom_1['clust'] = mb # creating a  new column and assigning it to new column 

Telecom_1.head()
Telecom_norm.head()

Telecom_1 = Telecom_1.iloc[:,[26,25,0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
Telecom_1.head()

Telecom3 =Telecom_1.iloc[:, 1:].groupby(Telecom_1.clust).mean()
os.chdir(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering")
Telecom_1.to_csv("Kmeans_Telecom.csv", encoding = "utf-8")
