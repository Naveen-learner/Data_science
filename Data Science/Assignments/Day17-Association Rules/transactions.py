# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:23:49 2022

@author: Naveen Kumar
"""

import pandas as pd # data manipulation
from mlxtend.frequent_patterns import apriori, association_rules 

transact = []
with open(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day17-Association Rules\transactions_retail.csv") as f: #open text files
    transact = f.read()

# splitting the data into separate transactions using separator as "\n"
transact = transact.split("\n")
transact3 = list(set(transact))

transact_list1 = []
for i in transact3:
    transact_list1.append(i.split(","))
    
transact_list = [[ subelt for subelt in elt if subelt != 'NA' ] for elt in transact_list1]

        
### Elemantary Analysis ###
all_transact_list = [i for item in transact_list for i in item] #extracting all items

from collections import Counter

item_frequencies = Counter(all_transact_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x: x[1]) #sorting according to frequencies

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))#ascending order
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

#plt.bar(height = frequencies[0:2621], x = list(range(0, 2621)), color = ['red','green','black','yellow','blue','pink','violet'])
plt.bar(height = frequencies[0:2621], x = list(range(0, 2621)), color = ['red','green','black','yellow','blue','pink','violet'])

plt.xticks(list(range(0, 2621), ), items[0:2621])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()
##########


# Creating Data Frame for the transactions data
transact_series = pd.DataFrame(pd.Series(transact_list))
transact_series = transact_series.iloc[:4227, :] # removing the last empty transaction

transact_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = transact_series['transactions'].str.join(sep = ',')
X = transact_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True) #len no of items in combo

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

# Association Rules
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part ###################################
# Handling Profusion of Rules (Duplication elimination)

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]# converting to tuples duplicates will be gone

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules10 = rules_no_redudancy.sort_values('lift', ascending = False).head(10)


