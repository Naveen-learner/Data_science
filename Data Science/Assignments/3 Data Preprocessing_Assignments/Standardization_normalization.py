# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:27:39 2022

@author: Naveen Kumar
"""
#import necessary libraries to do standardization and normalization
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import data file (Seeds_data.csv)
data = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\Seeds_data.csv")
### Standardization
# Initialise the Scaler
scaler = StandardScaler()

# To scale data
df = scaler.fit_transform(data)
# Convert the array back to a dataframe
dataset = pd.DataFrame(df)
res = dataset.describe()


### Normalization
## load data set
ethnic = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\Seeds_data.csv")
ethnic.columns

a1 = ethnic.describe()

# Get dummies will work only for categorical
#ethnic = pd.get_dummies(ethnic, drop_first = True)

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(ethnic)
b = df_norm.describe()
