# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:56:29 2022

@author: Naveen Kumar
"""
# import librariers
import pandas as pd
import numpy as np
import seaborn as sns

#import data file
data = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\boston_data.csv")
data.dtypes #to find which data type is the data_column
duplicate = data.duplicated() #To find duplicates
sum(duplicate) #To find sum of duplicates

#Plotting each data to check outliers
sns.boxplot(data.crim)
sns.boxplot(data.zn)
sns.boxplot(data.indus) #don't have outliers
sns.boxplot(data.chas)
sns.boxplot(data.nox)#don't have outliers
sns.boxplot(data.rm)
sns.boxplot(data.age)#don't have outliers
sns.boxplot(data.dis)
sns.boxplot(data.rad)#don't have outliers
sns.boxplot(data.tax)#don't have outliers
sns.boxplot(data.ptratio)
sns.boxplot(data.black)
sns.boxplot(data.lstat)
sns.boxplot(data.medv)

# Detection of outliers (find limits for salary based on IQR) and not considering which don't have outliers
IQR_1 = data['crim'].quantile(0.75) - data['crim'].quantile(0.25)
IQR_2 = data['zn'].quantile(0.75) - data['zn'].quantile(0.25)
IQR_3 = data['chas'].quantile(0.75) - data['chas'].quantile(0.25)
IQR_4= data['rm'].quantile(0.75) - data['rm'].quantile(0.25)
IQR_5 = data['dis'].quantile(0.75) - data['dis'].quantile(0.25)
IQR_6 = data['ptratio'].quantile(0.75) - data['ptratio'].quantile(0.25)
IQR_7 = data['black'].quantile(0.75) - data['black'].quantile(0.25)
IQR_8 = data['lstat'].quantile(0.75) - data['lstat'].quantile(0.25)
IQR_9 = data['medv'].quantile(0.75) - data['medv'].quantile(0.25)

lower_limit_1 = data['crim'].quantile(0.25) - (IQR_1 * 1.5)
upper_limit_1 = data['crim'].quantile(0.75) + (IQR_1 * 1.5)

lower_limit_2 = data['zn'].quantile(0.25) - (IQR_2 * 1.5)
upper_limit_2 = data['zn'].quantile(0.75) + (IQR_2 * 1.5)

lower_limit_3 = data['chas'].quantile(0.25) - (IQR_3 * 1.5)
upper_limit_3 = data['chas'].quantile(0.75) + (IQR_3 * 1.5)

lower_limit_4 = data['rm'].quantile(0.25) - (IQR_4 * 1.5)
upper_limit_4 = data['rm'].quantile(0.75) + (IQR_4 * 1.5)

lower_limit_5 = data['dis'].quantile(0.25) - (IQR_5 * 1.5)
upper_limit_5 = data['dis'].quantile(0.75) + (IQR_5 * 1.5)

lower_limit_6 = data['ptratio'].quantile(0.25) - (IQR_6 * 1.5)
upper_limit_6 = data['ptratio'].quantile(0.75) + (IQR_6 * 1.5)

lower_limit_7 = data['black'].quantile(0.25) - (IQR_7 * 1.5)
upper_limit_7 = data['black'].quantile(0.75) + (IQR_7 * 1.5)

lower_limit_8 = data['lstat'].quantile(0.25) - (IQR_8 * 1.5)
upper_limit_8 = data['lstat'].quantile(0.75) + (IQR_8 * 1.5)

lower_limit_9 = data['medv'].quantile(0.25) - (IQR_9 * 1.5)
upper_limit_9 = data['medv'].quantile(0.75) + (IQR_9 * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique(if we do trimming we cannot completely remove outliers as shown in below plots and removing the data doesn't make sense)
# Let's flag the outliers in the data set
'''outliers_data = np.where(data['crim'] > upper_limit_1, True, np.where(data['crim'] < lower_limit_1, True, False))
outliers_data = np.where(data['zn'] > upper_limit_2, True, np.where(data['zn'] < lower_limit_2, True, False))
outliers_data = np.where(data['chas'] > upper_limit_3, True, np.where(data['chas'] < lower_limit_3, True, False))
outliers_data = np.where(data['rm'] > upper_limit_4, True, np.where(data['rm'] < lower_limit_4, True, False))
outliers_data = np.where(data['dis'] > upper_limit_5, True, np.where(data['dis'] < lower_limit_5, True, False))
outliers_data = np.where(data['ptratio'] > upper_limit_6, True, np.where(data['ptratio'] < lower_limit_6, True, False))
outliers_data = np.where(data['black'] > upper_limit_7, True, np.where(data['black'] < lower_limit_7, True, False))
outliers_data = np.where(data['lstat'] > upper_limit_8, True, np.where(data['lstat'] < lower_limit_8, True, False))
outliers_data = np.where(data['medv'] > upper_limit_9, True, np.where(data['medv'] < lower_limit_9, True, False))
data_trimmed = data.loc[~(outliers_data), ]
data.shape, data_trimmed.shape
'''
outliers_data = np.where(data['crim'] > upper_limit_1, True, np.where(data['crim'] < lower_limit_1, True, False))
data_trimmed = data.loc[~(outliers_data), ]
data.shape, data_trimmed.shape

outliers_data_1 = np.where(data['zn'] > upper_limit_2, True, np.where(data['zn'] < lower_limit_2, True, False))
data_trimmed_1 = data.loc[~(outliers_data_1), ]
data.shape, data_trimmed_1.shape

outliers_data_2 = np.where(data['chas'] > upper_limit_3, True, np.where(data['chas'] < lower_limit_3, True, False))
data_trimmed_2 = data.loc[~(outliers_data_2), ]
data.shape, data_trimmed_2.shape

outliers_data_3 = np.where(data['rm'] > upper_limit_4, True, np.where(data['rm'] < lower_limit_4, True, False))
data_trimmed_3 = data.loc[~(outliers_data_3), ]
data.shape, data_trimmed_3.shape

outliers_data_4 = np.where(data['dis'] > upper_limit_5, True, np.where(data['dis'] < lower_limit_5, True, False))
data_trimmed_4 = data.loc[~(outliers_data_4), ]
data.shape, data_trimmed_4.shape

outliers_data_5 = np.where(data['ptratio'] > upper_limit_6, True, np.where(data['ptratio'] < lower_limit_6, True, False))
data_trimmed_5 = data.loc[~(outliers_data_5), ]
data.shape, data_trimmed_5.shape

outliers_data_6 = np.where(data['black'] > upper_limit_7, True, np.where(data['black'] < lower_limit_7, True, False))
data_trimmed_6 = data.loc[~(outliers_data_6), ]
data.shape, data_trimmed_6.shape

outliers_data_7 = np.where(data['lstat'] > upper_limit_8, True, np.where(data['lstat'] < lower_limit_8, True, False))
data_trimmed_7 = data.loc[~(outliers_data_7), ]
data.shape, data_trimmed_7.shape

outliers_data_8 = np.where(data['medv'] > upper_limit_9, True, np.where(data['medv'] < lower_limit_9, True, False))
data_trimmed_8 = data.loc[~(outliers_data_8), ]
data.shape, data_trimmed_8.shape

#Let's explore outliers in the trimmed dataset
sns.boxplot(data_trimmed.crim)
sns.boxplot(data_trimmed_1.zn)
sns.boxplot(data_trimmed_2.chas)
sns.boxplot(data_trimmed_3.rm)
sns.boxplot(data_trimmed_4.dis)
sns.boxplot(data_trimmed_5.ptratio)
sns.boxplot(data_trimmed_6.black)
sns.boxplot(data_trimmed_7.lstat)
sns.boxplot(data_trimmed_8.medv)
# We see no outliers(wrong)

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
data['data_crim'] = pd.DataFrame(np.where(data['crim'] > upper_limit_1, upper_limit_1, np.where(data['crim'] < lower_limit_1, lower_limit_1, data['crim'])))
sns.boxplot(data.data_crim)

data['data_zn'] = pd.DataFrame(np.where(data['zn'] > upper_limit_2, upper_limit_2, np.where(data['zn'] < lower_limit_2, lower_limit_2, data['zn'])))
sns.boxplot(data.data_zn)

data['data_chas'] = pd.DataFrame(np.where(data['chas'] > upper_limit_3, upper_limit_3, np.where(data['chas'] < lower_limit_3, lower_limit_3, data['chas'])))
sns.boxplot(data.data_chas)

data['data_rm'] = pd.DataFrame(np.where(data['rm'] > upper_limit_4, upper_limit_4, np.where(data['rm'] < lower_limit_4, lower_limit_4, data['rm'])))
sns.boxplot(data.data_rm)

data['data_dis'] = pd.DataFrame(np.where(data['dis'] > upper_limit_5, upper_limit_5, np.where(data['dis'] < lower_limit_5, lower_limit_5, data['dis'])))
sns.boxplot(data.data_dis)

data['data_ptratio'] = pd.DataFrame(np.where(data['ptratio'] > upper_limit_6, upper_limit_6, np.where(data['ptratio'] < lower_limit_6, lower_limit_6, data['ptratio'])))
sns.boxplot(data.data_ptratio)

data['data_black'] = pd.DataFrame(np.where(data['black'] > upper_limit_7, upper_limit_7, np.where(data['black'] < lower_limit_7, lower_limit_7, data['black'])))
sns.boxplot(data.data_black)

data['data_lstat'] = pd.DataFrame(np.where(data['lstat'] > upper_limit_8, upper_limit_8, np.where(data['lstat'] < lower_limit_8, lower_limit_8, data['lstat'])))
sns.boxplot(data.data_lstat)

data['data_medv'] = pd.DataFrame(np.where(data['medv'] > upper_limit_9, upper_limit_9, np.where(data['medv'] < lower_limit_9, lower_limit_9, data['medv'])))
sns.boxplot(data.data_medv)
# after rectifiying we can see there are no outliers present
############### 3. Winsorization ###############another outliers treatment method
#pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['crim','zn','chas','rm','dis','ptratio','black','lstat','medv'])

data_t = winsor.fit_transform(data[['crim','zn','chas','rm','dis','ptratio','black','lstat','medv']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t.crim)
sns.boxplot(data_t.zn)
sns.boxplot(data_t.chas)
sns.boxplot(data_t.rm)
sns.boxplot(data_t.dis)
sns.boxplot(data_t.ptratio)
sns.boxplot(data_t.black)
sns.boxplot(data_t.lstat)
sns.boxplot(data_t.medv)

