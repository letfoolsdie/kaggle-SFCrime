# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 22:51:01 2016

@author: Nikolay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import time
import datetime

df = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
seed = 42

le = LabelEncoder()
df['DayOfWeek'] = le.fit_transform(df.DayOfWeek)
test['DayOfWeek'] = le.transform(test.DayOfWeek)

df['PdDistrict'] = le.fit_transform(df.PdDistrict)
test['PdDistrict'] = le.transform(test.PdDistrict)

#Заменяем адреса отсутствующие в тестовой выборке на самое частое значение в обучающей: (исправить и заменить на ближайший адрес по координатам?)
test.loc[~test.Address.isin(np.unique(df.Address)),['Address']] = '800 Block of BRYANT ST'
df['Address'] = le.fit_transform(df.Address)
test['Address'] = le.transform(test.Address)

df['Dates'] = pd.to_datetime(df.Dates, yearfirst=True)
df['Year'] = df.Dates.dt.year
df['Month'] = df.Dates.dt.month
df['Hour'] = df.Dates.dt.hour

y = df.Category

features = ['DayOfWeek', 'PdDistrict','X','Y','Address', 'Year', 'Month', 'Hour']
cv = KFold(n=len(df), n_folds=5, shuffle=True, random_state=seed)

clf = LogisticRegression(random_state=seed)

start_time = datetime.datetime.now()
logloss = cross_val_score(clf, df[features], y, cv=cv, scoring='log_loss').mean()
duration = (datetime.datetime.now() - start_time).total_seconds()

print('Time elapsed:', duration)
print('average score (5 folds): %s' % logloss)
