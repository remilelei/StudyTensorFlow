# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf

testdata = pd.read_csv('test.csv')
testdata = testdata.fillna(0)
testdata['Sex'] = testdata['Sex'].apply(lambda s : 1 if s == 'male' else 0)

X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
