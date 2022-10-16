# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:45:43 2022

@author: Naveen Kumar
"""

###############################Hierarchial Clustering###############################
#importing necessary libraries for data manipulation, basic maths, visualtization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing datset using pandas
wine = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day14-PCA_Problem Statement\wine.csv")
wine.dtypes

duplicate = wine.duplicated()  # Return boolean Series denoting duplicate rows.
duplicate
sum(duplicate)
#checking outliers
wine.info()
sns.boxplot(wine.Type)
sns.boxplot(wine.Alcohol)
sns.boxplot(wine.Malic)
sns.boxplot(wine.Ash)
sns.boxplot(wine.Alcalinity)
sns.boxplot(wine.Magnesium)
sns.boxplot(wine.Phenols)
sns.boxplot(wine.Flavanoids)
sns.boxplot(wine.Nonflavanoids)
sns.boxplot(wine.Proanthocyanins)
sns.boxplot(wine.Color)
sns.boxplot(wine.Hue)
sns.boxplot(wine.Dilution)
sns.boxplot(wine.Proline)

############### 3. Winsorization ###############
pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Malic','Ash','Alcalinity','Magnesium','Proanthocyanins','Color','Hue'])

wine_t = winsor.fit_transform(wine[['Malic','Ash','Alcalinity','Magnesium','Proanthocyanins','Color','Hue']])
wine['Malic'] = wine_t['Malic']
wine['Ash'] = wine_t['Ash']
wine['Alcalinity'] = wine_t['Alcalinity']
wine['Magnesium'] = wine_t['Magnesium']
wine['Proanthocyanins'] = wine_t['Proanthocyanins']
wine['Color'] = wine_t['Color']
wine['Hue'] = wine_t['Hue']

wine.var() # variance of numeric variables
wine.var() == 0

# Check for count of NA's in each column
wine.isna().sum()

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

wine_norm = norm_func(wine.iloc[:,1:])
wine_norm.isna().sum()
b = wine_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram

z = linkage(wine_norm, method = "complete", metric = "euclidean")
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete', affinity = "euclidean").fit(wine_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
wine['Hierarical_clust'] = cluster_labels # creating a new column and assigning it to new column 
wine_norm1 = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine_norm1.iloc[:, 2:].groupby(wine_norm1.Hierarical_clust).mean()

#############################################K-Means Clustering###########################
###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine['K_Means_clust'] = mb # creating a  new column and assigning it to new column
wine_norm1 = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine_norm1.iloc[:, 2:].groupby(wine.K_Means_clust).mean()

##################PCA###############
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
wine_data = wine_norm.iloc[:, :]
pca = PCA(n_components = 3)
pca_values = pca.fit_transform(wine_data)

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
final = pd.concat([wine.Type, pca_data.iloc[:, 0:3]], axis = 1)
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))

final[['comp0', 'comp1', 'Type']].apply(lambda x: ax.text(*x), axis=1)

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
wine['Pca_clust1'] = cluster_labels # creating a new column and assigning it to new column 
wine_norm1 = wine.iloc[:,[16,15,14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine_norm1.iloc[:, 3:].groupby(wine_norm1.Pca_clust1).mean()
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
model = KMeans(n_clusters = 3)
model.fit(pca_data)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine['Pca_clust2'] = mb # creating a  new column and assigning it to new column
wine_norm1 = wine.iloc[:,[17,16,15,14,13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
wine_norm1.iloc[:, 5:].groupby(wine.Pca_clust2).mean()

#######Results are not same with and without clustering because of data loss after PCA######
#By using PCA you are losing information. If you do not want to lose too much, you can use as many PC as possible. (assume you can afford the computational efforts and there are not curse of dimensionality problem)

'''If you want to check how much 
information you lose, you can check my answers to 
PCA data before and after clustering to see how to get how much information (variance) preserved by PCA.'''