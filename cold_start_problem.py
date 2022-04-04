# prepare the pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib

from data_digest import AttractionsData
from surprise import KNNBasic

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Load our data set and compute the user similarity matrix
ml = AttractionsData()
data = ml.loadAttractionsDataLatestSmall()

trainSet = data.build_full_trainset()
y_train = trainSet.build_testset()
x_train = trainSet.build_anti_testset()

# tentar com este
# X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

sim_options = {'name': 'cosine',
               'user_based': True
               }

pipe = make_pipeline(StandardScaler(), KNNBasic(sim_options=sim_options))
pipe.fit(x_train)


joblib.dump(pipe, 'model.pkl')

## https://stackoverflow.com/questions/47416982/load-and-predict-new-data-sklearn
## https://stackoverflow.com/questions/38780302/predicting-new-data-using-sklearn-after-standardizing-the-training-data?rq=1