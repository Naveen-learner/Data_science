# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:15:13 2022

@author: Naveen Kumar
"""

import pandas as pd
import numpy as np
import seaborn as sns

glass = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day23-KNN\glass.csv")

glass.info()
glass['Type'] = glass.Type.astype('str')
b = glass.describe()
sns.boxplot(glass['RI'])#outliers present
sns.boxplot(glass['Na'])#outliers present
sns.boxplot(glass['Mg'])
sns.boxplot(glass['Al'])#outliers present
sns.boxplot(glass['Si'])#outliers present
sns.boxplot(glass['K'])#outliers present
sns.boxplot(glass['Ca'])#outliers present
sns.boxplot(glass['Ba'])#outliers present
sns.boxplot(glass['Fe'])#outliers present

glass.duplicated().sum()  # Return boolean Series denoting duplicate rows.
glass1 = glass.drop_duplicates() # Return DataFrame with duplicate rows remove
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['RI','Na','Al','Si','K','Ca','Ba','Fe'])

glass_t = winsor.fit_transform(glass1[['RI','Na','Al','Si','K','Ca','Ba','Fe']])

glass1['RI'] = glass_t['RI']
glass1['Na'] = glass_t['Na']
glass1['Al'] = glass_t['Al']
glass1['Si'] = glass_t['Si']
glass1['K'] = glass_t['K']
glass1['Ca'] = glass_t['Ca']
glass1['Ba'] = glass_t['Ba']
glass1['Fe'] = glass_t['Fe']

# If the variance is low or close to zero, then a feature is approximately 
# constant and will not improve the performance of the model.
# In that case, it should be removed. 

glass1.var() # variance of numeric variables
glass1.var() == 0
glass1.var(axis=0) == 0

glass1.drop(['RI','Ba','Fe'], axis = 1, inplace =True)

glass1.isna().sum()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

glass1.iloc[:,:6]

glass_n = norm_func(glass1.iloc[:, :6])
norm_data = glass_n.describe()

X = np.array(glass_n.iloc[:,:]) # Predictors 
Y = np.array(glass1['Type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Imbalance check
glass1.Type.value_counts()
glass1.Type.value_counts() / len(glass1.Type)  # values in percentages

ytrain = pd.DataFrame(Y_train)
ytest = pd.DataFrame(Y_test)

ytrain.value_counts() / len(ytrain)
ytest.value_counts() / len(ytest)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(1, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(1, 50, 2), [i[0] for i in acc],"ro-")  
# r = red color,  o = circle,  - = solid line

# test accuracy plot
plt.plot(np.arange(1, 50, 2), [i[1] for i in acc],"bo-") 
# b = blue color

