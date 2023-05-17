# Load the LogisticRegression classifier
# Note, use CV for cross-validation as requested in the question
from sklearn.linear_model import ElasticNetCV

# Load some other sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
# Import other libraries
import pandas as pd, numpy as np
import sys
import os
from matplotlib import pyplot as plt

percent_pc = 0.95
X = pd.read_csv(f'esm1b.inference/esm.{sys.argv[1]}.training.tokens.csv', index_col=0)
os.makedirs(f'elastic.net.result/{sys.argv[1]}.{sys.argv[2]}/', exist_ok=True)

# do SVD on X before training
Xu, Xs, Xvt = np.linalg.svd(X)
def calc_k(S, percentge):
  '''
  identify the minimum k value to reconstruct
  '''
  k = 0
  total = sum(np.square(S))
  svss = 0 #singular values square sum
  for i in range(np.shape(S)[0]):
      svss += np.square(S[i])
      if (svss/total) >= percentge:
          k = i+1
          break
  return k

k = calc_k(Xs, percent_pc) # get the number of k to reconstruct 0.9 square sum

def buildSD(S, k):
  '''
  reconstruct k singular value diag matrix
  '''
  SD = np.eye(k) * S[:k]
  return SD

def buildSD_1(S, k):
  '''
  reconstruct k singular value diag matrix
  '''
  SD = np.eye(k) * 1/S[:k]
  return SD


Xu_new = pd.DataFrame(Xu[:len(Xu), :k], index=X.index)
Xvt_new = Xvt[:k, :len(Xvt)]
Xs_new = buildSD(Xs, k)
Xs_new_1 = buildSD_1(Xs, k)
X_new = pd.DataFrame(np.dot(np.dot(Xu_new, Xs_new), Xvt_new), index=X.index)

labels = pd.read_csv(f'itan/esm.{sys.argv[1]}.training.csv')
y = labels[f'score.{sys.argv[2]}']
y = y.astype(float)

X_train, y_train = X_new, y

X_test = pd.read_csv(f'esm1b.inference/esm.{sys.argv[1]}.testing.tokens.csv', index_col=0)
labels_test = pd.read_csv(f'itan/esm.{sys.argv[1]}.testing.csv')
y_test = labels_test[f'score.{sys.argv[2]}']
y_test = y_test.astype(float)

elastic_net_regressor = ElasticNetCV(cv=5, l1_ratio=np.linspace(0.01, 1, 21), tol=0.0001, max_iter=10000, n_jobs=64)

elastic_net_regressor.fit(X_train, y_train)

#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

with open(f'elastic.net.result/{sys.argv[1]}.{sys.argv[2]}/elastic.net.report.txt', 'w') as outputfile:
  outputfile.write('Training MSE: {}\n'.format(mean_squared_error(y_train, elastic_net_regressor.predict(X_train))))
  outputfile.write('Testing MSE: {}\n'.format(mean_squared_error(y_test, elastic_net_regressor.predict(X_test))))
  outputfile.write("Training R2: {}\n".format(r2_score(y_train, elastic_net_regressor.predict(X_train))))
  outputfile.write("Testing R2: {}".format(r2_score(y_test, elastic_net_regressor.predict(X_test))))


import pickle
with open(f'elastic.net.result/{sys.argv[1]}.{sys.argv[2]}/elasticnet.pkl', 'wb') as outputfile:
  pickle.dump(elastic_net_regressor, outputfile)
  
prediction=labels_test
#prediction['is.train']=np.isin(prediction['genes'], X_train.index.values)
prediction['prediction']=elastic_net_regressor.predict(X_test)
# prediction['prediction_prob']=elastic_net_regressor.predict_proba(X_test)[:,0]
prediction.to_csv(f'elastic.net.result/{sys.argv[1]}.{sys.argv[2]}/prediction.test.csv')

# draw fpr tpr
# fpr, tpr, thresholds = metrics.roc_curve(y_test.astype(float),
#                                          elastic_net_regressor.predict_proba(X_test)[:,1],
#                                          pos_label=1)
# plt.plot(fpr, tpr, label="AUC={:.3f}".format(metrics.auc(fpr, tpr)))
# plt.xticks(np.arange(0.0, 1.1, step=0.1))
# plt.xlabel("False Positive Rate", fontsize=15)
# plt.yticks(np.arange(0.0, 1.1, step=0.1))
# plt.ylabel("True Positive Rate", fontsize=15)
# plt.legend(prop={'size': 13}, loc='lower right')
# plt.show()
# plt.savefig(f'elastic.net.result/{sys.argv[1]}.{sys.argv[2]}/auc.test.pdf')
# plt.close()


all_prediction = pd.DataFrame(elastic_net_regressor.predict(X_train), index=labels.index)
all_prediction.to_csv(f'elastic.net.result/{sys.argv[1]}.{sys.argv[2]}/prediction.train.csv')
# PC_loadings = pd.read_csv(f'{sys.argv[1]}/PC.loadings.csv', index_col=0)
# Xs_new_Xvt_new = np.dot(Xs_new, Xvt_new)
# Xs_new_Xvt_new_1 = np.dot(Xvt_new.T, Xs_new_1)
# elastic_net_loadings = pd.DataFrame(np.dot(Xs_new_Xvt_new_1, elastic_net_regressor.coef_.T), index=X.columns.values)
elastic_net_loadings = pd.DataFrame(elastic_net_regressor.coef_.T, index=X.columns.values)
elastic_net_loadings.to_csv(f'elastic.net.result/{sys.argv[1]}.{sys.argv[2]}/elastic.net.loadings.csv')
# elastic_net_intercept = pd.DataFrame(elastic_net_regressor.intercept_)
# elastic_net_intercept.to_csv(f'elastic.net.result/{sys.argv[1]}.{sys.argv[2]}/elastic.net.intercept.csv')


