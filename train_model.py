# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:10:04 2018

@author: Frank
"""

from data_digest import AttractionsData
from surprise import KNNBasic, SVD
import pickle
from surprise.model_selection import cross_validate
# from recommender_metrics import RecommenderMetrics
        
# Load our data set and compute the user similarity matrix
ml = AttractionsData()
data = ml.loadAttractionsDataLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
# model = SVD()
model.fit(trainSet)
class ModelObj:
    model = model
    trainSet = trainSet
    ml = ml
modelObj = ModelObj()

cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True, verbose=True)
        
# Save model
with open('model.pickle', 'wb') as f:
    pickle.dump(modelObj, f)
    