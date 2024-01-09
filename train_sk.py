#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/09 

import numpy as np
from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score

X_FILE = 'output/X.npy'
Y_FILE = 'output/Y.npy'

X = np.load(X_FILE)
Y = np.load(Y_FILE)

model = LogisticRegressionCV(cv=10, class_weight='balanced', verbose=True)
model.fit(X, Y)
pred = model.predict(X)
print(classification_report(Y, pred))
print('Acc:', accuracy_score(Y, pred))
