# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:56:12 2022

@author: Naveen Kumar
"""
#importing libraries
import pandas as pd

#importing csv file
games = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\DataSets\game.csv", encoding = 'utf8')

games.shape # shape
games.columns


games.duplicated().sum()  # Return boolean Series denoting duplicate rows.
games.isna().sum()

from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf = TfidfVectorizer()
# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(games.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 47

# calculating the dot product using sklearn's linear_kernel()
from sklearn.metrics.pairwise import linear_kernel

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of anime name to index number 
games_index = pd.Series(games.index, index = games['userId']).drop_duplicates()

userId = games_index[53]
userId

def get_recommendations(userId, topN):    
    # topN = 10
    # Getting the movie index using its title 
    userId = games_index[userId]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[userId]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    user_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    games_similar_show = pd.DataFrame(columns=["name", "Score"])
    games_similar_show["name"] = games.loc[user_idx, "game"]
    games_similar_show["Score"] = game_scores
    games_similar_show.reset_index(inplace = True)  
    print (games_similar_show)
    # The End

    
# Enter your anime and number of anime's to be recommended
games_index[66]
get_recommendations(66, topN = 10)

