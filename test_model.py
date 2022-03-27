# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:10:04 2018

@author: Frank
"""

import pickle
import heapq
from collections import defaultdict
from operator import itemgetter
from train_model import ModelObj

def getPredictions(k, testSubject):     
    simsMatrix = modelObj.model.compute_similarities()

    # Get top N similar users to our test subject
    testUserInnerID = modelObj.trainSet.to_inner_uid(testSubject)
    similarityRow = simsMatrix[testUserInnerID]

    similarUsers = []
    for innerID, score in enumerate(similarityRow):
        if (innerID != testUserInnerID):
            similarUsers.append( (innerID, score) )

    kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

    # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
    candidates = defaultdict(float)
    for similarUser in kNeighbors:
        innerID = similarUser[0]
        userSimilarityScore = similarUser[1]
        theirRatings = modelObj.trainSet.ur[innerID]
        for rating in theirRatings:
            candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
        
    # Build a dictionary of stuff the user has already seen
    watched = {}
    for itemID, rating in modelObj.trainSet.ur[testUserInnerID]:
        watched[itemID] = 1
        
    # Get top-rated items from similar users:
    print('\n', "TOP 5 dos lugares mais recomendados para o usuário:", testSubject, '\n')
    pos = 0
    results = []
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            attractionID = modelObj.trainSet.to_raw_iid(itemID)
            results.append(modelObj.ml.getAttractionName(int(attractionID)))
            print(results[pos], ratingSum)
            pos += 1
            if (pos > 4):
                break
    
    return results

# Load model
with open('model.pickle', 'rb') as f:
    modelObj = pickle.load(f)