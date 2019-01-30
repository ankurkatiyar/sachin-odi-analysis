import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

sachinDF = pd.read_csv('sachin_odi_latest.csv')
print(sachinDF.head())

sachinDF.info()

corr = sachinDF.corr()
print(corr)
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)

plt.show()

dfTrain = sachinDF[:]

trainLabel = np.asarray(dfTrain['Won'])
trainData = np.asarray(dfTrain.drop('Won',1))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)

print means
print stds

trainData = (trainData - means)/stds

print trainData

odiResultCheck = LogisticRegression()
odiResultCheck.fit(trainData, trainLabel)

coeff = list(odiResultCheck.coef_[0])
print coeff

labels = list(dfTrain.drop('Won',1).columns)
print labels

print pd.DataFrame()

features = pd.DataFrame()
features['Attributes'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Attributes', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')

plt.show()

joblib.dump([odiResultCheck, means, stds], 'odiResultModel.pkl')
