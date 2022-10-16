# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:37:28 2022

@author: Naveen Kumar
"""

import pandas as pd

# import Dataset 
movies = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\Entertainment.csv", encoding = 'utf8')
movies.shape # shape
movies.columns
movies.Category # genre columns

from sklearn.feature_extraction.text import TfidfVectorizer
# term frequency inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(movies.Titles)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 47

# From the above matrix we need to find the similarity score.
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 
# cosine(x,y)= (x.y⊺)/(||x||.||y||)

# calculating the dot product using sklearn's linear_kernel()
from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
# creating a mapping of anime name to index number 
movies_Id = pd.Series(movies.index,index=movies['Titles'])
movies_Id.index=movies['Titles']
movies_id = movies_Id['Grumpier Old Men (1995)']
movies_id

def get_recommendations(name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    movies_id = movies_Id[name]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[movies_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    movies_idx  =  [i[0] for i in cosine_scores_N]
    movies_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    movies_similar_show = pd.DataFrame(columns=["name", "Score"])
    movies_similar_show["name"] = movies.loc[movies_idx, "Titles"]
    movies_similar_show["Score"] = movies_scores
    movies_similar_show.reset_index(inplace = True)  
    print (movies_similar_show)
    # The End

    
# Enter your anime and number of anime's to be recommended
movies_Id['Grumpier Old Men (1995)']
get_recommendations('Grumpier Old Men (1995)', topN = 10)

