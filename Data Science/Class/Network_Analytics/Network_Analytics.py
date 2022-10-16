import pandas as pd
import networkx as nx 

# Load the dataset
G = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Class\Network_Analytics\routes.csv")
G = G.iloc[0:500, 1:10]
G.info()

g = nx.Graph()
g = nx.from_pandas_edgelist(G, source = 'Source Airport', target = 'Destination Airport')

print(nx.info(g))



# Update the decorator package to 5.0.7 to over come the 'random_state_index is incorrect' error
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

