# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:51:09 2022

@author: Naveen Kumar
"""

import numpy as np 
import pandas as pd
import collections
import networkx as nx

# Read route data
route_cols = ["flights","ID", "main Airport", "main Airport ID", "Destination","Destination ID","Codeshare","haults","machinary"]
routes_df = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\connecting_routes.csv",skiprows=1,names = route_cols)
routes_df['main Airport ID'] = pd.to_numeric(routes_df['main Airport ID'].astype(str), 'coerce')
routes_df['Destination ID'] = pd.to_numeric(routes_df['Destination ID'].astype(str), 'coerce')
    
print(routes_df.shape)
routes_df.head()

# Read airport data
airport_col = ["ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time"]
airport_df = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\flight_hault.csv",skiprows=1,names = airport_col)
print(airport_df.shape)
airport_df.tail()

# Drop airport that don't have IATA data
routes_df = routes_df[routes_df.flights != '\\N']
routes_df = routes_df[routes_df.ID != '\\N']
routes_df = routes_df[routes_df['main Airport'] != '\\N']
routes_df = routes_df[routes_df['main Airport ID'] != '\\N']
routes_df = routes_df[routes_df['Destination'] != '\\N']
routes_df = routes_df[routes_df['Destination ID'] != '\\N']
routes_df = routes_df[routes_df['Codeshare'] != '\\N']
routes_df = routes_df[routes_df['haults'] != '\\N']
routes_df = routes_df[routes_df['machinary'] != '\\N']

airport_df = airport_df[airport_df.ID != '\\N']
airport_df = airport_df[airport_df.Name != '\\N']
airport_df = airport_df[airport_df.City != '\\N']
airport_df = airport_df[airport_df.Country != '\\N']
airport_df = airport_df[airport_df['IATA_FAA'] != '\\N']
airport_df = airport_df[airport_df.ICAO != '\\N']
airport_df = airport_df[airport_df.Latitude != '\\N']
airport_df = airport_df[airport_df.Longitude != '\\N']
airport_df = airport_df[airport_df.Altitude != '\\N']
airport_df = airport_df[airport_df.Time != '\\N']
airport_df = airport_df[airport_df.DST != '\\N']
airport_df = airport_df[airport_df["Tz database time"] != '\\N']
print(airport_df.shape)
airport_df.tail()

# make new route df with route count info
routes_all = pd.DataFrame(routes_df.groupby(['main Airport', 'Destination']).size().reset_index(name='counts'))

airport_all = airport_df[['Name','City','Country','Latitude', 'Longitude', 'IATA_FAA','ICAO']]
IATA_array = airport_all["IATA_FAA"].tolist()
ICAO_array = airport_all["ICAO"].tolist()

# only keep route with airport have IATA code
routes_all = routes_all[routes_all['main Airport'].isin(IATA_array)]
routes_all = routes_all[routes_all['Destination'].isin(ICAO_array)]
routes_all.duplicated().sum()
# Create networkX graph
G = routes_all.iloc[0:5,:]
g = nx.Graph()
g = nx.from_pandas_edgelist(G, source = 'main Airport', target = 'Destination', edge_attr = 'counts',create_using = nx.DiGraph())
print(nx.info(g))

 #Update the decorator package to 5.0.7 to over come the 'random_state_index is incorrect' error
# pip install decorator==5.0.7

pos = nx.spring_layout(g)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')

# Degree Centrality
d = nx.degree_centrality(g)
print(d) 

# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)

# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(g) 
print(cc)
