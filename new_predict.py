# prepare the pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from data_digest import AttractionsData
from surprise import KNNBasic

from new_dataset import createNewData
from pandas import pd

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

#Loading the saved model with joblib
pipe = joblib.load('model.pkl')

# make new dataset
# createNewData()

# New data to predict
pr = pd.read_csv('datas/new_data.csv')
pred_cols = list(pr.columns.values)[:-1]

# scaler transformer
pipe.transform(pr[pred_cols])

# model predict
# apply the whole pipeline to data
pred = pd.Series(pipe.predict(pr[pred_cols]))
print(pred)