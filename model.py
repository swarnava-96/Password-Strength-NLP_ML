# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 02:28:14 2021

@author: SWARNAVA
"""

# CHECKING PASSWORD STRENGTH USING NLP-ML


## Importing necessary libraries
import numpy as np
import pandas as pd
import random
import seaborn as sns

## Loading the dataset
data = pd.read_csv('C:/Users/SWARNAVA/Desktop/password/data.csv',',',error_bad_lines = False)
data.head()
data.shape

## Feature Engineering
### Handling null values
data.isnull().sum()
data[data['password'].isnull()] ##Identifying the null value
data.dropna(inplace=True) ##Dropping the null value

password_tuple = np.array(data)  ##Converting into array
password_tuple

random.shuffle(password_tuple)  ##Shuffling randomly for robustness

## Dividing into dependent and independent feature
y = [labels[1] for labels in password_tuple]
y

X = [labels[0] for labels in password_tuple]
X

sns.set_style('whitegrid')
sns.countplot(x = 'strength', data=data, palette='RdBu_r')

def word_divide_char(inputs):  ##For extracting each letter of a word
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=word_divide_char)
X = vectorizer.fit_transform(X)


X.shape

data.iloc[0,0] ##Checking the first value

## Logistics Regression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  #splitting

# ovr
log_class=LogisticRegression(penalty='l2',multi_class='ovr')
log_class.fit(X_train,y_train)

print(log_class.score(X_test,y_test))

## Multinomial

clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
clf.fit(X_train, y_train) #training
print(clf.score(X_test, y_test))



# XGBoost Classifier

import xgboost as xgb

xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(X_train,y_train)

xgb_classifier.score(X_test,y_test)

# MultinomialNB

from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()

nb_classifier.fit(X_train,y_train)
nb_classifier.score(X_test,y_test)

