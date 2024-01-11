#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/09 

import numpy as np
from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score
from utils import load_truth, load_db, OUT_PATH

# Acc: 69.68140%
#RESULT_FILE = OUT_PATH / 'stats_aekl_mse.json'
# Acc: 71.32580%
#RESULT_FILE = OUT_PATH / 'stats_aekl_ema.json'
# Acc: 71.83967%
#RESULT_FILE = OUT_PATH / 'stats_aekl_ema_sample.json'
# Acc: 71.83967%
#RESULT_FILE = OUT_PATH / 'stats_aekl_ema_noise-randn_1e-3.json'
# Acc: 71.83967%
#RESULT_FILE = OUT_PATH / 'stats_aekl_ema_noise-randu_1e-3.json'
# Acc: 72.04522%
#RESULT_FILE = OUT_PATH / 'stats_aekl_ema_noise-randu_1e-1.json'
# Acc: 69.37307%
RESULT_FILE = OUT_PATH / 'stats_aekl_ema_noise-randu_1.json'

preds = list(load_db(RESULT_FILE).values())
X = np.stack(preds, axis=0)
Y = np.asarray(load_truth())
assert len(X) == len(Y), f'>> len {len(X)} != {len(Y)}'

model = LogisticRegressionCV(cv=10, class_weight='balanced', verbose=False)
model.fit(X, Y)
pred = model.predict(X)
print(classification_report(Y, pred))
print(f'Accuracy: {accuracy_score(Y, pred):.5%}')
