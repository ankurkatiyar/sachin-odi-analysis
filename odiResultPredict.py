import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

odiLoadedModel, means, stds = joblib.load('odiResultModel.pkl')
labels = ['Runs','SR','Pos','Inns','Won']
sampleData = [(100,0,1,2,1)]

sampleData = pd.DataFrame.from_records(sampleData, columns=labels)
# prepare sample
print sampleData

sampleDataFeatures = np.asarray(sampleData.drop('Won',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds

# predict
predictionProbability = odiLoadedModel.predict_proba(sampleDataFeatures)
prediction = odiLoadedModel.predict(sampleDataFeatures)
print('ODI Result Probability:', predictionProbability)
print('ODI Result Prediction:', prediction)
