# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:24:38 2022

@author: Naveen Kumar
"""


#import necessary libraries
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
import numpy as np

#Read excel using pandas
Insurance = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day12-Hierarchical Clustering\Autoinsurance.csv")

Insurance_1 = Insurance.drop(['Customer','State'],axis =1 )
Insurance_1.describe()
Insurance_1.info()

duplicate = Insurance_1.duplicated()  # Return boolean Series denoting duplicate rows.
sum(duplicate) #There are no duplicates found
Insurance1 = Insurance_1.drop_duplicates() # Return DataFrame with duplicate rows removed.

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

Insurance1_t = winsor.fit_transform(Insurance_1[['Customer Lifetime Value','Monthly Premium Auto','Number of Open Complaints','Number of Policies','Total Claim Amount']])
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


#################### Missing Values - Imputation ###########################
import numpy as np
import pandas as pd

# Check for count of NA's in each column
Insurance1.isna().sum()

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
from sklearn.preprocessing import OneHotEncoder
# Creating instance of One-Hot Encoder
enc = OneHotEncoder() # initializing method
Insurance1.columns
Insurance1 = Insurance1.iloc[:,[1,4,5,6,9,14,15,16,17,20,0,2,3,7,8,10,11,12,13,18,19]]
Insurance1 = Insurance1[['Response', 'Effective To Date', 'EmploymentStatus', 'Gender',
       'Marital Status', 'Policy Type', 'Policy', 'Renew Offer Type',
       'Sales Channel', 'Vehicle Size', 'Customer Lifetime Value', 'Coverage',
       'Education', 'Income', 'Location Code', 'Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Policies', 'Total Claim Amount', 'Vehicle Class']]

 = pd.DataFrame(enc.fit_transform(Insurance1.iloc[:, 0:9]).toarray())

#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()

# df['desig'] = labelencoder.fit_transform(df['Position'])

# Data Split into Input and Output variables
#X = df.iloc[:, :9]
#y = df.iloc[:, 9]

Insurance1['Coverage'] = labelencoder.fit_transform(Insurance1['Coverage'])
Insurance1['Education'] = labelencoder.fit_transform(Insurance1['Education'])
Insurance1['Location Code'] = labelencoder.fit_transform(Insurance1['Location Code'])
Insurance1['Vehicle Class'] = labelencoder.fit_transform(Insurance1['Vehicle Class'])
#Insurance1 = pd.get_dummies(Insurance1, drop_first = True)


# Created dummies for all categorical columns

### Normalization
### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

Insurance_norm = norm_func(Insurance1)
b = Insurance_norm.describe()
Insurance_norm.isna().sum()


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram

z = linkage(Insurance_norm, method = "complete", metric = "euclidean")
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(Insurance_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
Insurance_norm['clust'] = cluster_labels # creating a new column and assigning it to new column 
Insurance_norm.dropna(inplace = True)
np.arange(0,104,1)
Insurance_norm1 = Insurance_norm.iloc[:, [104,0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103]] #re-order columns
Insurance_norm1.head()
Insurance_norm1.iloc[:, :]
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
