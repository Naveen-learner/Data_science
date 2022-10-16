# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:28:03 2022

@author: Naveen Kumar
"""
#import libraries
import pandas as pd
import numpy as np
import seaborn as sns

# Load the dataset
# Use the modified Claimants dataset
data = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\Claimants.csv")

# Check for count of NA's in each column
data.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (Salaries)
# Mode is used for discrete data (ex: Position, Sex, MaritalDesc)

# For Mean, Median, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Median Imputer because of the claimant's age
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(data[["CLMAGE"]]))
data["CLMAGE"].isna().sum()  # all records replaced by median 

data.isna().sum()

# Mode Imputer(As there only 0 and 1 values)
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data["CLMSEX"] = pd.DataFrame(mode_imputer.fit_transform(data[["CLMSEX"]]))
data["CLMINSUR"] = pd.DataFrame(mode_imputer.fit_transform(data[["CLMINSUR"]]))
data["SEATBELT"] = pd.DataFrame(mode_imputer.fit_transform(data[["SEATBELT"]]))
data.isnull().sum()  # all Sex, MaritalDesc records replaced by mode
