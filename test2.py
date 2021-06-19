# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:19:42 2021

@author: Aman Bhanse
"""

import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("raw_data_1.csv")
df.dtypes


#find unique values
nf = df['PCURRENTCOMMAND'].unique()
nf = pd.DataFrame(nf)
nf2 = df['PPREVIOUSCOMMAND'].unique()
nf2 = pd.DataFrame(nf2)

nf3 = pd.concat([nf , nf2])
nf3 = nf3[0].unique()
nf3 = pd.DataFrame(nf3)

#HashingEncoding
from sklearn.feature_extraction import FeatureHasher

h= FeatureHasher(n_features = 4 , input_type='string')

result = h.fit_transform(df["PCURRENTCOMMAND"])
result = result.toarray()


x = df.iloc[: , :-1]
y= df.iloc[: ,-1 ]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder() , [4] )] , remainder='passthrough')
ct.fit_transform(df)

from sklearn.neighbors import KNeighborsClassifier
cf = KNeighborsClassifier(n_neighbors =5 , metric = 'minkowski' , p=2)

cf.fit(x,y)

item_similarity = cosine_similarity(df) 