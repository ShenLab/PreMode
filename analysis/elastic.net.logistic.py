# Load the LogisticRegression classifier
# Note, use CV for cross-validation as requested in the question
from sklearn.linear_model import LogisticRegressionCV

# Load some other sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
# Import other libraries
import pandas as pd, numpy as np
import sys
import os
import yaml
from matplotlib import pyplot as plt

# $1 is the dataset name
# $2 is the yaml directory to compare with
percent_pc = 0.95
# load dataset
X = pd.read_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/esm2.inference/{sys.argv[2]}/training.fold.{sys.argv[3]}.tokens.csv', index_col=0)
os.makedirs(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/', exist_ok=True)
# get train & validation in deep learning training
with open(f'{sys.argv[1]}/{sys.argv[2]}.5fold/{sys.argv[2]}.fold.{sys.argv[3]}.yaml', 'r') as f:
    data = yaml.safe_load(f)
train_idx = np.load(f"{data['log_dir']}/splits.0.npz")['idx_train']
train_df = pd.read_csv(f"{data['data_file_train']}", index_col=0).iloc[train_idx]
y_columns = [col for col in train_df.columns if col.startswith('score')]
X = X.iloc[train_idx]
# remove the score-0,3 samples
if len(y_columns) == 1:
  X = X.loc[(train_df['score'].values != 0) & (train_df['score'].values != 3)]
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

y_columns = [col for col in train_df.columns if col.startswith('score')]
if len(y_columns) == 1:
  labels = train_df[(train_df['score'].values != 0) & (train_df['score'].values != 3)]
  y = labels['score']
  y = y.astype(int)
  X_train, y_train = X_new, y
else:
  labels = train_df
  X_train = X_new

X_test = pd.read_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/esm2.inference/{sys.argv[2]}/testing.fold.{sys.argv[3]}.tokens.csv', index_col=0)
labels_test = pd.read_csv(f"{data['data_file_test']}", index_col=0)
# do for all scores
prediction=labels_test

if len(y_columns) > 1:
  for y_column in y_columns:
    y_test = labels_test[y_column]
    y_test = y_test.astype(int)
    y_train = labels[y_column].astype(int)

    elastic_net_classifier = LogisticRegressionCV(cv=5, penalty='elasticnet', l1_ratios=np.linspace(0, 1, 21),
                                                  random_state=0,
                                                  solver='saga', tol=0.0001, max_iter=10000, n_jobs=64)

    elastic_net_classifier.fit(X_train, y_train)

    #prediction['is.train']=np.isin(prediction['genes'], X_train.index.values)
    prediction[f'prediction.{y_column}']=elastic_net_classifier.predict(X_test)
    prediction[f'prediction_prob.{y_column}']=elastic_net_classifier.predict_proba(X_test)[:,0]
    prediction.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/prediction.test.fold.{sys.argv[3]}.csv')
else:
  # remove the score-0 samples
  X_test = X_test.iloc[(labels_test['score'].values != 0) & (labels_test['score'].values != 3)]
  labels_test = labels_test.iloc[(labels_test['score'].values != 0) & (labels_test['score'].values != 3)]
  y_test = labels_test['score']
  y_test = y_test.astype(int)

  elastic_net_classifier = LogisticRegressionCV(cv=5, penalty='elasticnet', l1_ratios=np.linspace(0, 1, 21),
                                                random_state=0,
                                                solver='saga', tol=0.0001, max_iter=10000, n_jobs=64)

  elastic_net_classifier.fit(X_train, y_train)

  #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

  with open(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/elastic.net.report.fold.{sys.argv[3]}.txt', 'w') as outputfile:
    outputfile.write('Training Summary:\n')
    outputfile.write(classification_report(y_train, elastic_net_classifier.predict(X_train)))
    outputfile.write('\nTesting Summary:\n')
    outputfile.write(classification_report(y_test, elastic_net_classifier.predict(X_test)))
    outputfile.write('\n')
    outputfile.write("Training accuracy: {}\n\n".format(elastic_net_classifier.score(X_train, y_train)))
    outputfile.write("Testing accuracy: {}".format(elastic_net_classifier.score(X_test, y_test)))


  import pickle
  with open(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/elasticnet.fold.{sys.argv[3]}.pkl', 'wb') as outputfile:
    pickle.dump(elastic_net_classifier, outputfile)
    
  prediction=labels_test
  #prediction['is.train']=np.isin(prediction['genes'], X_train.index.values)
  prediction['prediction']=elastic_net_classifier.predict(X_test)
  prediction['prediction_prob']=elastic_net_classifier.predict_proba(X_test)[:,0]
  prediction.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/prediction.test.fold.{sys.argv[3]}.csv')

  # draw fpr tpr
  fpr, tpr, thresholds = metrics.roc_curve(y_test.astype(float),
                                          elastic_net_classifier.predict_proba(X_test)[:,1],
                                          pos_label=1)
  plt.plot(fpr, tpr, label="AUC={:.3f}".format(metrics.auc(fpr, tpr)))
  plt.xticks(np.arange(0.0, 1.1, step=0.1))
  plt.xlabel("False Positive Rate", fontsize=15)
  plt.yticks(np.arange(0.0, 1.1, step=0.1))
  plt.ylabel("True Positive Rate", fontsize=15)
  plt.legend(prop={'size': 13}, loc='lower right')
  plt.show()
  plt.savefig(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/auc.test.fold.{sys.argv[3]}.pdf')
  plt.close()


  all_prediction = pd.DataFrame(elastic_net_classifier.predict_proba(X_train), index=labels.index)
  all_prediction.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/prediction.train.fold.{sys.argv[3]}.csv')
  # PC_loadings = pd.read_csv(f'{sys.argv[2]}/PC.loadings.csv', index_col=0)
  # Xs_new_Xvt_new = np.dot(Xs_new, Xvt_new)
  # Xs_new_Xvt_new_1 = np.dot(Xvt_new.T, Xs_new_1)
  # elastic_net_loadings = pd.DataFrame(np.dot(Xs_new_Xvt_new_1, elastic_net_classifier.coef_.T), index=X.columns.values)
  elastic_net_loadings = pd.DataFrame(elastic_net_classifier.coef_.T, index=X.columns.values)
  elastic_net_loadings.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/elastic.net.loadings.fold.{sys.argv[3]}.csv')
  elastic_net_intercept = pd.DataFrame(elastic_net_classifier.intercept_)
  elastic_net_intercept.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/elastic.net.intercept.fold.{sys.argv[3]}.csv')


