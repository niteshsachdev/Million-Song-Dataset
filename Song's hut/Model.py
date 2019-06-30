# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:25:37 2019

@author: Nitesh Sachdev
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Clean_MSD.csv')
features = pd.read_csv('Features_List.csv').drop('Unnamed: 0',axis=1)
labels = dataset['song_hotttnesss']
scores = []

#Splitting labels and features
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3,random_state =0)

#Handling Data imbalance using Smote
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
features_train, labels_train = sm.fit_sample(features_train, labels_train.ravel())


#Scaling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test)  

import pickle
with open('scaling.pkl','wb') as f:
    pickle.dump(sc,f)

#XGB approach
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(features_train,labels_train)
labels_pred = model.predict(features_test)

# evaluate predictions
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#Checking If data imbalance is resolved
labels_pred = pd.DataFrame(labels_pred)[0].value_counts()

#Applying Cross validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(model,features_test,labels_test, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

#Creating Pickle file
import pickle
with open('model.pkl','wb') as f:
    pickle.dump(model,f)


"""
#DecisionTree approach

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test)
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)
#RandomForest approach

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(features_train, labels_train)  
labels_pred = classifier.predict(features_test)
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)


#SVM approach
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 7)
classifier.fit(features_train, labels_train)
# Predicting the Test set results
labels_pred = classifier.predict(features_test)
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#NaiveBayes approach
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
gnb = GaussianNB()
gnb.fit(features_train,labels_train)
labels_pred = gnb.predict(features_test)
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)
# Predicting the class labels
labels_pred = classifier.predict(features_test)
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#K-nn Approach
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2)
classifier.fit(features_train, labels_train)
# Predicting the class labels
labels_pred = classifier.predict(features_test)
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#Plotting the scores
approach = ['XGB','DecisionTree','RandomForest','SVM','NB','LR','KNN']
import matplotlib.pyplot as plt
plt.bar(approach,scores)

"""
