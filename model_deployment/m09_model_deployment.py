#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os
os.chdir('..')
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder


# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)

dataTotal= pd.concat([dataTraining,dataTesting], axis=0)
enc = OrdinalEncoder()
dataTotal[['State','Make','Model']] = enc.fit_transform(dataTotal[['State','Make','Model']])

X=dataTotal.iloc[:400000,:].drop(['Price'], axis=1)
y=dataTraining['Price']

XTest=dataTotal.iloc[400000:,:].drop(['Price'], axis=1)

clf = XGBRegressor(max_depth=10, n_estimators=100, gamma=0, learning_rate=0.2,random_state=1)
clf.fit(X, y)


joblib.dump(clf, 'Price_Car_Grupo4.pkl', compress=3)


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(url)
        
        print(url)
        print('Probability of Phishing: ', p1)
        