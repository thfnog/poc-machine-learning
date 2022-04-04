# -*- coding: utf-8 -*-

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

# recommender_metrics.
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True, verbose=True)
        
# Save model
with open('model.pickle', 'wb') as f:
    pickle.dump(modelObj, f)
    
    
# benchmark = []
# # Iterate over all algorithms
# for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
#     # Perform cross validation
#     results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    
#     # Get results & append algorithm name
#     tmp = pd.DataFrame.from_dict(results).mean(axis=0)
#     tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
#     benchmark.append(tmp)
    
# pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 