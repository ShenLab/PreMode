# Load the LogisticRegression classifier
# Note, use CV for cross-validation as requested in the question
from sklearn.ensemble import GradientBoostingClassifier

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
# $2 is the test data file name
def transform_df(df):
  for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')
  # Convert categorical terms to integers
  for col in df.columns:
      if pd.api.types.is_categorical_dtype(df[col]):
          df[col] = df[col].cat.codes
  return df

X_train = pd.read_csv(f'{sys.argv[1]}', index_col=0)
X_train = transform_df(X_train)
# y_train = pd.read_csv(f'{sys.argv[2]}', index_col=0)
y_train = X_train['Class'].astype(int)
X_train.drop(columns=['Class'], inplace=True)
X_test = pd.read_csv(f'{sys.argv[2]}', index_col=0)
X_test = transform_df(X_test)
# y_test = pd.read_csv(f'{sys.argv[4]}', index_col=0)
y_test = X_test['Class'].astype(int)
X_test.drop(columns=['Class'], inplace=True)
# Identify columns in df2 that are not in df
columns_to_drop = [col for col in X_test.columns if col not in X_train.columns]
# Drop columns from df2
X_test.drop(columns=columns_to_drop, inplace=True)

gbm_classifier = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)

# draw fpr tpr
fpr, tpr, thresholds = metrics.roc_curve(y_test.astype(float),
                                        gbm_classifier.predict_proba(X_test)[:,1],
                                        pos_label=1)
print("AUC={:.9f}".format(metrics.auc(fpr, tpr)))


