# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:46:46 2022

@author: Naveen Kumar
"""

import pandas as pd # data manipulation
from mlxtend.frequent_patterns import apriori, association_rules 

movies = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Day17-Association Rules\my_movies.csv")

movies = movies.iloc[:,5:]
from collections import Counter

item_frequencies = Counter(movies)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x: x[1]) #sorting according to frequencies

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))#ascending order
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:10], x = list(range(0, 10)), color = ['red','green','black','yellow','blue','pink','violet'])
plt.bar(height = frequencies[0:10], x = list(range(0, 10)), color = ['red','green','black','yellow','blue','pink','violet'])

plt.xticks(list(range(0, 10), ), items[0:10])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()
##########

frequent_itemsets = apriori(movies, min_support = 0.0075, max_len = 4, use_colnames = True) #len no of items in combo

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
