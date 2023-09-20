# Load the LogisticRegression classifier
# Note, use CV for cross-validation as requested in the question
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
# Load some other sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import sys
# Import other libraries
import pandas as pd
import numpy as np
import os

# $1 is the train file
# $2 is the test file
X_train_file = pd.read_csv(f'{sys.argv[1]}', index_col=0)
X_test_file = pd.read_csv(f'{sys.argv[2]}', index_col=0)
X_train_file = X_train_file.dropna(subset=["uniprotID"])
X_test_file = X_test_file.dropna(subset=["uniprotID"])
# X_train_file = X_train_file[X_train_file['score'].values != 0]
X_test_file = X_test_file[X_test_file['score'].values != 0]
# os.makedirs(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/', exist_ok=True)
# get train & validation in deep learning training
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoder.fit(np.array([X_train_file["uniprotID"].values.tolist() + X_test_file["uniprotID"].values.tolist()]).T)
X_train = one_hot_encoder.transform(np.array([X_train_file["uniprotID"].values.tolist()]).T)
X_test = one_hot_encoder.transform(np.array([X_test_file["uniprotID"].values.tolist()]).T)

y_train = X_train_file['score']
y_test = X_test_file['score']
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# X_train, y_train = , 
# prediction=X_test

rf_classifier = RandomForestClassifier(random_state=0).fit(X_train, y_train)
# gp_classifier.fit(X_train, y_train)
# import pickle
# with open(f'{sys.argv[1]}/gp.{seed}.pkl', 'wb') as outputfile:
#   pickle.dump(rf_classifier, outputfile)

#prediction['is.train']=np.isin(prediction['genes'], X_train.index.values)
# prediction[f'prediction']=rf_classifier.predict(X_test)
# prediction[f'prediction_prob']=rf_classifier.predict_proba(X_test)[:,0]

# draw fpr tpr
# import ipdb; ipdb.set_trace()
y_test_logits = rf_classifier.predict_proba(X_test)
# if (y_test_logits.shape[1] == 3):
#     y_test_logits = y_test_logits[:,2] / (y_test_logits[:,0] + y_test_logits[:,2])
# else:
#     y_test_logits = y_test_logits[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test.astype(float),
                                         y_test_logits[:,-1],
                                         pos_label=1)
# plt.plot(fpr, tpr, label="AUC={:.3f}".format(metrics.auc(fpr, tpr)))
# plt.xticks(np.arange(0.0, 1.1, step=0.1))
# plt.xlabel("False Positive Rate", fontsize=15)
# plt.yticks(np.arange(0.0, 1.1, step=0.1))
# plt.ylabel("True Positive Rate", fontsize=15)
# plt.legend(prop={'size': 13}, loc='lower right')
# plt.show()
# plt.savefig(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/auc.test.{seed}.pdf')
# plt.close()

# with open(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/GP.report.{seed}.txt', 'w') as outputfile:
#   outputfile.write('Training Summary:\n')
#   outputfile.write(classification_report(y_train, rf_classifier.predict(X_train)))
#   outputfile.write('\nTesting Summary:\n')
#   outputfile.write(classification_report(y_test, rf_classifier.predict(X_test)))
#   outputfile.write('\n')
#   outputfile.write("Training accuracy: {}\n\n".format(rf_classifier.score(X_train, y_train)))
#   outputfile.write("Testing accuracy: {}\n\n".format(rf_classifier.score(X_test, y_test)))
print("Testing AUC: {}".format(metrics.auc(fpr, tpr)))

# prediction.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/prediction.csv')
# all_prediction = pd.DataFrame(gp_classifier.predict_proba(X_train), index=labels.index)
# all_prediction.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/prediction.train.csv')
# PC_loadings = pd.read_csv(f'{sys.argv[1]}/PC.loadings.csv', index_col=0)
# Xs_new_Xvt_new = np.dot(Xs_new, Xvt_new)
# Xs_new_Xvt_new_1 = np.dot(Xvt_new.T, Xs_new_1)
# elastic_net_loadings = pd.DataFrame(np.dot(Xs_new_Xvt_new_1, elastic_net_classifier.coef_.T), index=X.columns.values)
# elastic_net_loadings = pd.DataFrame(gp_classifier.coef_.T, index=X.columns.values)
# elastic_net_loadings.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/elastic.net.loadings.csv')
# elastic_net_intercept = pd.DataFrame(gp_classifier.intercept_)
# elastic_net_intercept.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/elastic.net.intercept.csv')


