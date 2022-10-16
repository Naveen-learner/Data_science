# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:52:55 2022

@author: Naveen Kumar
"""

import pandas as pd
import networkx as nx 
import numpy as np

# Load the dataset
G = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\facebook.csv")
H = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\instagram.csv")
I = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\linkedin.csv")

#create a network matrix using adjacency matrix
fb = np.matrix(G)
f = nx.from_numpy_matrix(fb)
nx.draw(f)

ig = np.matrix(H)
i = nx.from_numpy_matrix(ig)
nx.draw(i)

ln = np.matrix(I)
l = nx.from_numpy_matrix(ln)
nx.draw(i)


