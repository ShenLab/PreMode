# Load the LogisticRegression classifier
# Note, use CV for cross-validation as requested in the question
from sklearn.ensemble import RandomForestClassifier

# Load some other sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
# Import other libraries
import pandas as pd, numpy as np
import os
import yaml
import sys
# $1 is the train data file name
# $2 is the train label file name
# $3 is the test data file name
# $4 is the test data file name
X_train = pd.read_csv(f'{sys.argv[1]}', index_col=0)
y_train = pd.read_csv(f'{sys.argv[2]}', index_col=0)
y_train = y_train['score']
y_train = y_train.astype(int)
X_test = pd.read_csv(f'{sys.argv[3]}', index_col=0)
y_test = pd.read_csv(f'{sys.argv[4]}', index_col=0)
y_test = y_test['score']
y_test = y_test.astype(int)

gp_classifier = RandomForestClassifier(random_state=0).fit(X_train, y_train)

# draw fpr tpr
fpr, tpr, thresholds = metrics.roc_curve(y_test.astype(float),
                                        gp_classifier.predict_proba(X_test)[:,1],
                                        pos_label=1)
print("AUC={:.9f}".format(metrics.auc(fpr, tpr)))


