# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 09:21:49 2022

@author: Naveen Kumar
"""

##############Import necessaary librariers#############
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
##################Import dataset file#################

df = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\data-cleaning-rmotr-freecodecamp-master\data\btc-eth-prices-outliers.csv")
df.dtypes

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business 
#Central tendency for bitcoin data
df.Bitcoin.mean()
df.Bitcoin.median()
df.Bitcoin.mode()

#Central tendency for Ether data
df.Ether.mean()
df.Ether.mode()
df.Ether.mode()

# Measures of Dispersion / Second moment business decision for Bitcoin
df.Bitcoin.var()
df.Bitcoin.std()
range = max(df.Bitcoin) - min(df.Bitcoin)
range

# Measures of Dispersion / Second moment business decision for Ether
df.Ether.var()
df.Ether.std()
range = max(df.Ether) - min(df.Ether)
range

# Third moment business decision
df.Bitcoin.skew()
df.Ether.skew()

# Fourth moment business decision
df.Bitcoin.kurt()
df.Bitcoin.kurt()

#Data Visualization
sns.boxplot(df.Bitcoin)
sns.boxplot(df.Ether)
plt.hist(df.Bitcoin)
plt.hist(df.Ether)
df.plot()

######################Data Pre-Processing######################
#Typecasting
#No need of typecasting for this data

#Handling duplicates
duplicate = df.duplicated()
sum(duplicate)
#####There are no duplicates found

####Missing Values Imputation because for winsorization need to remove all Nan values
df.isna().sum()

######Mean or Median Imputation because of numerical values
# For Mean, Median, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

###mean Imputation
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Ether"] = pd.DataFrame(mean_imputer.fit_transform(df[["Ether"]]))
df["Ether"].isna().sum()

################################################
############## Outlier Treatment ###############
############### 3. Winsorization ###############
#pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Bitcoin','Ether'])

df_t = winsor.fit_transform(df[['Bitcoin','Ether']])

############Data visualization after winsorization
sns.boxplot(df_t.Bitcoin)
sns.boxplot(df_t.Ether)
########## Change Columns of original data
df['Bitcoin'] = df_t['Bitcoin']
sns.boxplot(df.Bitcoin)
df['Ether'] = df_t['Ether']
sns.boxplot(df.Ether)

##############################################
#### zero variance and near zero variance ####

# If the variance is low or close to zero, then a feature is approximately 
# constant and will not improve the performance of the model.
# In that case, it should be removed. 

df.var() # variance of numeric variables
df.var() == 0
df.var(axis=0) == 0

############## Discretization ####################
df.dtypes
df.head()
df.info()
df.describe()

df['Bitcoin_new'] = pd.cut(df['Bitcoin'], 
                              bins = [min(df.Bitcoin),
                                      df.Bitcoin.mean(),
                                      max(df.Bitcoin)],
                              labels=["Low", "High"])

df['Ether_new'] = pd.cut(df['Ether'], 
                              bins = [min(df.Ether),
                                      df.Ether.mean(),
                                      max(df.Ether)],
                              labels=["Low", "High"])

df.head()
df.Bitcoin_new.value_counts()
df.Ether_new.value_counts()
df.isna().sum()

##################################################
################## Dummy Variables ###############No need to create dummies because of all numerical data

df.drop(['Timestamp'], axis = 1, inplace = True)

# Create dummy variables
df_new = pd.get_dummies(df)

#####################
# Normal Quantile-Quantile Plot
# Checking Whether data is normally distributed
stats.probplot(df.Bitcoin, dist="norm", plot=pylab)

stats.probplot(df.Ether, dist="norm", plot=pylab)

stats.probplot(np.sqrt(df.Bitcoin), dist = 'norm', plot = pylab)
stats.probplot(np.sqrt(df.Ether), dist = 'norm', plot = pylab)

### Standardization
from sklearn.preprocessing import StandardScaler
df.drop(['Timestamp'], axis = 1, inplace = True) #######As timestamp is object

# Initialise the Scaler
scaler = StandardScaler()

# To scale data
data = scaler.fit_transform(df)
# Convert the array back to a dataframe
dataset = pd.DataFrame(data)
res = dataset.describe()


### Normalization
df.drop(['Timestamp'], axis = 1, inplace = True) #######As timestamp is object


### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(df)
b = df_norm.describe()





