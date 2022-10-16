# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:34:21 2022

@author: Naveen Kumar
"""

import pandas as pd
import numpy as np
import seaborn as sns

zoo = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day23-KNN\zoo.csv")

zoo.info()
zoo.describe()

b = zoo.describe()

sns.boxplot(zoo['hair'])
sns.boxplot(zoo['feathers'])#outliers present
sns.boxplot(zoo['eggs'])
sns.boxplot(zoo['milk'])
sns.boxplot(zoo['airborne'])#outliers present
sns.boxplot(zoo['predator'])
sns.boxplot(zoo['toothed'])
sns.boxplot(zoo['backbone'])#outliers present
sns.boxplot(zoo['breathes'])#outliers present
sns.boxplot(zoo['venomous'])#outliers present
sns.boxplot(zoo['fins'])#outliers present
sns.boxplot(zoo['legs'])#outliers present
sns.boxplot(zoo['tail'])
sns.boxplot(zoo['domestic'])#outliers present
sns.boxplot(zoo['catsize'])

zoo.duplicated().sum()

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['feathers','airborne','backbone','breathes','venomous','fins','legs','domestic'])

zoo_t = winsor.fit_transform(zoo[['feathers','airborne','backbone','breathes','venomous','fins','legs','domestic']])

zoo['feathers'] = zoo_t['feathers']
zoo['airborne'] = zoo_t['airborne']
zoo['backbone'] = zoo_t['backbone']
zoo['breathes'] = zoo_t['breathes']
zoo['venomous'] = zoo_t['venomous']
zoo['fins'] = zoo_t['fins']
zoo['legs'] = zoo_t['legs']
zoo['domestic'] = zoo_t['domestic']

# If the variance is low or close to zero, then a feature is approximately 
# constant and will not improve the performance of the model.
# In that case, it should be removed. 

zoo.var() # variance of numeric variables
zoo.var() == 0
zoo.var(axis=0) == 0

zoo.drop(['animal name','feathers','airborne','backbone','breathes','venomous','fins','domestic'], axis = 1, inplace =True)
zoo.isna().sum()
zoo.iloc[:,:9]
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

zoo_n = norm_func(zoo.iloc[:, :9])
norm_data = zoo_n.describe()

X = np.array(zoo_n.iloc[:,:]) # Predictors 
Y = np.array(zoo['type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Imbalance check
zoo['type'].value_counts()
zoo['type'].value_counts() / len(zoo['type'])  # values in percentages

ytrain = pd.DataFrame(Y_train)
ytest = pd.DataFrame(Y_test)

ytrain.value_counts() / len(ytrain)
ytest.value_counts() / len(ytest)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
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

for i in range(1, 10, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(1, 10, 2), [i[0] for i in acc],"ro-")  
# r = red color,  o = circle,  - = solid line

# test accuracy plot
plt.plot(np.arange(1, 10, 2), [i[1] for i in acc],"bo-") 
# b = blue color

