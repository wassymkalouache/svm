#binary classifier: 2 groups at a time
#it will separate groups into one at a time
#find the best separating hyperplan: 2 dimensional space, at most 3 dim
#once you acquire it, you can take unknown datapoints: see which side it is in

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop([' Class'], 1))
y = np.array(df[' Class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)

