# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:24:09 2021

@author: Aman Bhanse
"""
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("dataset.csv")
data = data.fillna(0)

#Normalization

def standardize(row):
    new_row = (row-row.mean())/(row.max() - row.min())
    return new_row


data_nor = data.apply(standardize)

#item-item similarity so i need to transpose the dataset since item need to be in y axis

item_similarity = cosine_similarity(data_nor.T) #this is symmetic matix
item_similarity_df = pd.DataFrame(item_similarity , index = data.columns , columns = data.columns)


#for recommendations
def get_similar_movies(movie_name , user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating - 2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

print(get_similar_movies("Romantic1",1))

