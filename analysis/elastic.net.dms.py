# Load the LogisticRegression classifier
# Note, use CV for cross-validation as requested in the question
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
# Load some other sklearn functions
from sklearn import metrics
# Import other libraries
import pandas as pd, numpy as np
import os
import yaml
import sys
from scipy.stats import spearmanr
# $1 is the train data file name
# $2 is the train label file name
# $3 is the test data file name
# $4 is the test data file name
# $5 is optional, test result
X_train = pd.read_csv(f'{sys.argv[1]}', index_col=0)
y_train = pd.read_csv(f'{sys.argv[2]}', index_col=0)
score_columns = y_train.columns[y_train.columns.str.startswith('score')]
y_train = y_train[score_columns]

X_test = pd.read_csv(f'{sys.argv[3]}', index_col=0)
y_test = pd.read_csv(f'{sys.argv[4]}', index_col=0)
y_test = y_test[score_columns]

regr = MultiTaskElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], random_state=0, n_jobs=64, max_iter=10000).fit(X_train, y_train)
# draw fpr tpr
y_pred = regr.predict(X_test)
for i, col in enumerate(y_test.columns):
    res = spearmanr(y_test[col], y_pred[:, i]).correlation
    print("Rho={:.9f}".format(res))

# If an output file is specified, save y_pred to it
if len(sys.argv) > 5:
    pd.DataFrame(y_pred, index=X_test.index, columns=score_columns).to_csv(sys.argv[5])
