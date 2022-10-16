# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:57:23 2022

@author: Naveen Kumar
"""

#import necessary libraries
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
import numpy as np

#Read excel using pandas
Telecom = pd.read_excel(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering\Telco_customer_churn.xlsx")

Telecom_1 = Telecom.drop(['Customer ID'],axis =1 )
Telecom_1.describe()
Telecom_1.info()

duplicate = Telecom_1.duplicated()  # Return boolean Series denoting duplicate rows.
sum(duplicate) #There are no duplicates found


#To check outliers
sns.boxplot(Telecom_1['Count'])
sns.boxplot(Telecom_1['Number of Referrals'])#outliers present
sns.boxplot(Telecom_1['Tenure in Months'])
sns.boxplot(Telecom1['Bonus_trans'])
sns.boxplot(Telecom_1['Avg Monthly Long Distance Charges'])
sns.boxplot(Telecom_1['Avg Monthly GB Download'])#outliers present
sns.boxplot(Telecom_1['Monthly Charge'])
sns.boxplot(Telecom_1['Total Charges'])
sns.boxplot(Telecom_1['Total Refunds'])#outliers present
sns.boxplot(Telecom_1['Total Extra Data Charges'])#outliers present
sns.boxplot(Telecom_1['Total Long Distance Charges'])#outliers present
sns.boxplot(Telecom_1['Total Revenue'])#outliers present

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

'''Discretizzation
Telecom1['Balance_new'] = pd.cut(Telecom1['Balance'], #pandas.qcut for more number of bins
                              bins = [min(Telecom1.Balance),
                                      Telecom1.Balance.mean(),
                                      max(Telecom1.Balance)],
                              labels=["zero", "one"])

Telecom1['Bonus_miles_new'] = pd.cut(Telecom1['Bonus_miles'], 
                              bins = [min(Telecom1.Bonus_miles),
                                      Telecom1.Bonus_miles.mean(),
                                      max(Telecom1.Bonus_miles)],
                              labels=["zero", "one"])
Telecom1['Bonus_trans_new'] = pd.cut(Telecom1['Bonus_trans'], 
                              bins = [min(Telecom1.Bonus_trans),
                                      Telecom1.Bonus_trans.mean(),
                                      max(Telecom1.Bonus_trans)],
                              labels=["zero", "one"])
Telecom1['Flight_miles_12mo_new'] = pd.cut(Telecom1['Flight_miles_12mo'], 
                              bins = [min(Telecom1.Flight_miles_12mo),
                                      Telecom1.Flight_miles_12mo.mean(),
                                      max(Telecom1.Flight_miles_12mo)],
                              labels=["zero", "one"])
Telecom1['Flight_trans_12_new'] = pd.cut(Telecom1['Flight_trans_12'], 
                              bins = [min(Telecom1.Flight_trans_12),
                                      Telecom1.Flight_trans_12.mean(),
                                      max(Telecom1.Flight_trans_12)],
                              labels=["zero", "one"])
Telecom1['Days_since_enroll_new'] = pd.cut(Telecom1['Days_since_enroll'], 
                              bins = [min(Telecom1.Days_since_enroll),
                                      Telecom1.Days_since_enroll.mean(),
                                      max(Telecom1.Days_since_enroll)],
                              labels=["zero", "one"])
Telecom1['Award?_new'] = pd.cut(Telecom1['Award?'], 
                              bins = [min(Telecom1['Award?']),
                                      Telecom1['Award?'].mean(),
                                      max(Telecom1['Award?'])],
                              labels=["zero", "one"])
'''
#################### Missing Values - Imputation ###########################
import numpy as np
import pandas as pd

# Check for count of NA's in each column
Telecom_1.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (Salaries)
# Mode is used for discrete data (ex: Position, Sex, MaritalDesc)

# For Mean, Median, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

'''Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
Telecom1["Balance_new"] = pd.DataFrame(mode_imputer.fit_transform(Telecom1[["Balance_new"]]))
Telecom1["Bonus_miles_new"] = pd.DataFrame(mode_imputer.fit_transform(Telecom1[["Bonus_miles_new"]]))
Telecom1["Bonus_trans_new"] = pd.DataFrame(mode_imputer.fit_transform(Telecom1[["Bonus_trans_new"]]))
Telecom1["Flight_miles_12mo_new"] = pd.DataFrame(mode_imputer.fit_transform(Telecom1[["Flight_miles_12mo_new"]]))
Telecom1["Flight_trans_12_new"] = pd.DataFrame(mode_imputer.fit_transform(Telecom1[["Flight_trans_12_new"]]))
Telecom1["Days_since_enroll_new"] = pd.DataFrame(mode_imputer.fit_transform(Telecom1[["Days_since_enroll_new"]]))
Telecom1["Award?_new"] = pd.DataFrame(mode_imputer.fit_transform(Telecom1[["Award?_new"]]))
Telecom1.isna().sum()  # all Sex, MaritalDesc records replaced by mode
'''

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



# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram

z = linkage(Telecom_norm, method = "complete", metric = "euclidean")
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(Telecom_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
Telecom_norm['clust'] = cluster_labels # creating a new column and assigning it to new column 

Telecom_norm1 = Telecom_norm.iloc[:, [33,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]] #re-order columns
Telecom_norm1.head()
Telecom_norm1.iloc[:, :]
# Aggregate mean of each cluster for labellling the cluster
#only clustering is success when we are able label it
Telecom_3 = Telecom_norm1.iloc[:, 1:].groupby(Telecom_norm1.clust).mean() #groupping by mean clustering
Telecom_norm1.iloc[:, 1:].groupby(Telecom_norm1.clust).std() #groupping by std clustering

Telecom_3['churn_rate'] = np.where(Telecom_3['Total Revenue']>= 0.25, True, False)

Telecom_norm1['Total Revenue'].groupby(Telecom_norm1.clust).mean() #groupping by mean clustering
Telecom_norm1['Total Revenue'].groupby(Telecom_norm1.clust).std() #groupping by std clustering


# creating a csv file 
Telecom_norm1.to_csv("Churn_rate.csv", encoding = "utf-8")


os.getcwd() #get current working directory
os.chdir(r"C:\Users\Naveen Kumar\Desktop\Data Science\Class\Datasets")
