# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 18:40:31 2022

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

airport_all = airport_df[['Name','City','Country','Latitude', 'Longitude', 'IATA_FAA']]
IATA_array = airport_all["IATA_FAA"].tolist()

# extract japan airport info
airport_jp = airport_df[(airport_df.Country == "Japan")][['ID','Name','City','Latitude','Longitude','IATA_FAA']]
#jp_airport_ix = airport_jp.index.values
routes_jp = routes_df[(routes_df['main Airport ID'].isin(airport_jp['ID'])) &
                      (routes_df['Destination ID'].isin(airport_jp['ID']))] 

routes_all.head()

# only keep route with airport have IATA code
routes_all = routes_all[routes_all['main Airport'].isin(IATA_array)]
routes_all = routes_all[routes_all['Destination'].isin(IATA_array)]

# add route for all 2 airports in same city

# make 2 temp df

local_source_ap = airport_all[['City','Country','IATA_FAA']].copy()
local_source_ap.rename({'IATA_FAA': 'main Airport'}, axis=1, inplace=True)
local_source_ap.dropna(inplace=True)

local_dest_ap = airport_all[['City','Country','IATA_FAA']].copy()
local_dest_ap.rename({'IATA_FAA': 'Destination'}, axis=1, inplace=True)
local_dest_ap.dropna(inplace=True)

print(local_source_ap.shape)

# only consider airpot that already have routes

# make set of all airport with route
ap_set1 = set(routes_all["main Airport"].tolist())
ap_set2 = set(routes_all["Destination"].tolist())
print(len(ap_set1))
print(len(ap_set2))
ap_set1.update(ap_set2)
print(len(ap_set1))

local_source_ap2 = local_source_ap[(local_source_ap['main Airport'].isin(ap_set1))]
local_dest_ap2 = local_dest_ap[(local_dest_ap['Destination'].isin(ap_set1))]

print(local_source_ap2.shape)
print(local_dest_ap2.shape)

s1 = set(local_source_ap2['main Airport'].tolist())
s2 = set(local_dest_ap2['Destination'].tolist())
print(s1.difference(s2))
local_route = pd.merge(local_source_ap2, local_dest_ap2, how='inner', on=['City', 'Country'])
local_route.rename({'main Airport': 'Source'}, axis=1, inplace=True)
local_route = local_route.query("Source != Destination")

print(local_route.shape)
local_route
routes_all.rename({'main Airport': 'Source'}, axis=1, inplace=True)
interset = pd.merge(local_route, routes_all, how='inner', on=['Source', 'Destination'])
interset

routes_all_n_local = routes_all.append(local_route)
print(routes_all_n_local.shape)
routes_all_n_local.drop(['City', 'Country'], axis=1, inplace=True)
routes_all_n_local['counts'] = routes_all_n_local['counts'].fillna(1)
routes_all_n_local.head()

# to find number of flights in and out of an airport
# it is similar to find number of rows in which each airport occur in either one of the 2 columns
counts = routes_all['Source'].append(routes_all.loc[routes_all['Source'] != routes_all['Destination'], 'Destination']).value_counts()

# create a data frame of position based on names in count
counts = pd.DataFrame({'IATA_FAA': counts.index, 'total_flight': counts})
pos_data = counts.merge(airport_all, on = 'IATA_FAA')

routes_100 = routes_all.nlargest(100, 'counts')
routes_100.head()

# Create networkX graph
G = routes_all.iloc[0:100,:]
g = nx.Graph()
g = nx.from_pandas_edgelist(G, source = 'Source', target = 'Destination', edge_attr = 'counts',create_using = nx.DiGraph())
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
