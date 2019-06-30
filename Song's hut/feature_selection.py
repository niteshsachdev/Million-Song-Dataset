# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:38:22 2019

@author: Nitesh Sachdev
"""
#Importing the dataset using Pandas
import pandas as pd
import numpy as np
dataset = pd.read_csv('Clean_MSD.csv').drop('Unnamed: 0',axis=1)

#Checking for null values in the dataset
l1 = {}
for item in dataset:
    l1[item] = dataset[item].isnull().value_counts()
#Temprory feature variable 
features = dataset[['artist_familiarity','artist_hotttnesss','artist_id','artist_latitude','artist_location','artist_longitude','artist_name','duration','end_of_fade_in','key','key_confidence','loudness','mode','mode_confidence','release','start_of_fade_out','tempo','time_signature','time_signature_confidence','title','year']]
features.info()
features = features.drop(['artist_location', 'artist_latitude', 'artist_longitude','artist_name', 'release', 'title','mode_confidence'],axis = 1)
labels = dataset['song_hotttnesss']

#Label Encoding of the temprory features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
features['artist_id'] = features['artist_id'].astype(str)
features['artist_id'] = le.fit_transform(features['artist_id'])
features['artist_familiarity'] = features['artist_familiarity'].fillna(features['artist_familiarity'].mean())
"""
mode = []
for i, row in features.iterrows():
    if features['mode'][i] == 1:
        mode.append(features['mode_confidence'][i])
    else:
        mode.append(-features['mode_confidence'][i])
    print(mode[i])
features = features.drop(['mode_confidence','mode'],axis=1)
features['mode'] = mode
"""

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(features,labels)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances.nlargest(16).plot(kind='barh')
plt.show()

features = features.drop('time_signature',axis=1)
features.to_csv('Features_List.csv')