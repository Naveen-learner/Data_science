# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:31:07 2022

@author: Naveen Kumar
"""
###############################Hierarchial Clustering###############################
#importing necessary libraries for data manipulation, basic maths, visualtization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing datset using pandas
disease = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day14-PCA_Problem Statement\heart disease.csv")
disease.dtypes

duplicate = disease.duplicated()  # Return boolean Series denoting duplicate rows.
duplicate
sum(duplicate)
# Removing Duplicates
disease1 = disease.drop_duplicates() # Return DataFrame with duplicate rows removed.
#checking outliers
disease1.info()
sns.boxplot(disease1.age)
sns.boxplot(disease1.sex)
sns.boxplot(disease1.cp)
sns.boxplot(disease1.trestbps)
sns.boxplot(disease1.chol)
sns.boxplot(disease1.fbs)
sns.boxplot(disease1.restecg)
sns.boxplot(disease1.thalach)
sns.boxplot(disease1.exang)
sns.boxplot(disease1.oldpeak)
sns.boxplot(disease1.slope)
sns.boxplot(disease1.ca)
sns.boxplot(disease1.thal)
sns.boxplot(disease1.target)

############### 3. Winsorization ###############
pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['trestbps','chol','fbs','thalach','oldpeak','ca','thal'])

disease_t = winsor.fit_transform(disease1[['trestbps','chol','fbs','thalach','oldpeak','ca','thal']])
disease1['trestbps'] = disease_t['trestbps']
disease1['chol'] = disease_t['chol']
disease1['fbs'] = disease_t['fbs']
disease1['thalach'] = disease_t['thalach']
disease1['oldpeak'] = disease_t['oldpeak']
disease1['ca'] = disease_t['ca']
disease1['thal'] = disease_t['thal']

disease1.var() # variance of numeric variables
disease1.var() == 0

disease1.drop(['fbs'], axis = 1, inplace =True)

# Check for count of NA's in each column
disease1.isna().sum()

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

disease_norm = norm_func(disease1)
disease_norm.isna().sum()
b = disease_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram

z = linkage(disease_norm, method = "complete", metric = "euclidean")
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete', affinity = "euclidean").fit(disease_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
disease1['clust'] = cluster_labels # creating a new column and assigning it to new column 
disease_norm1 = disease1.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
disease_norm1.iloc[:, 1:].groupby(disease_norm1.clust).mean()

#############################################K-Means Clustering###########################
###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(disease_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(disease_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
disease1['K_Means_clust'] = mb # creating a  new column and assigning it to new column
disease_norm1 = disease1.iloc[:,[14,13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
disease_norm1.iloc[:, 1:].groupby(disease1.K_Means_clust).mean()

##################PCA###############
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
disease_data = disease_norm.iloc[:, :]
pca = PCA(n_components = 3)
pca_values = pca.fit_transform(disease_data)

# PCA weights
pca.components_
pca.components_[0]

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2"
final = pd.concat([disease1.target, pca_data.iloc[:, 0:3]], axis = 1)
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))

final[['comp0', 'comp1', 'target']].apply(lambda x: ax.text(*x), axis=1)

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram

z_1 = linkage(pca_data, method = "complete", metric = "euclidean")
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z_1, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete', affinity = "euclidean").fit(pca_data) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
disease1['Pca_clust'] = cluster_labels # creating a new column and assigning it to new column 
disease_norm1 = disease1.iloc[:,[16,15,14,13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
disease_norm1.iloc[:, 4:].groupby(disease_norm1.Pca_clust).mean()
#############################K-Means###############################
from sklearn.cluster import	KMeans
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(pca_data)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(pca_data)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
disease1['Pca_k_clust'] = mb # creating a  new column and assigning it to new column
disease_norm1 = disease1.iloc[:,[17,16,15,14,13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
disease_norm1.iloc[:, 4:].groupby(disease1.Pca_k_clust).mean()

#######Results are not same with and without clustering because of data loss after PCA######
'''If you want to check how much 
information you lose, you can check my answers to 
PCA data before and after clustering to see how to get how much information (variance) preserved by PCA.'''