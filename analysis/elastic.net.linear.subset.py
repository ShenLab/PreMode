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
import yaml
from matplotlib import pyplot as plt

percent_pc = 0.95
# load dataset
X = pd.read_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/esm2.inference/{sys.argv[2]}/training.{sys.argv[3]}.{sys.argv[4]}.tokens.csv', index_col=0)
os.makedirs(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/', exist_ok=True)
with open(f'{sys.argv[1]}/{sys.argv[2]}.subsets/subset.{sys.argv[3]}/seed.{sys.argv[4]}.yaml', 'r') as f:
    data = yaml.safe_load(f)
train_idx = np.load(f"{data['log_dir']}/splits.0.npz")['idx_train']
train_df = pd.read_csv(f"{data['data_file_train']}", index_col=0).iloc[train_idx]

X = X.iloc[train_idx]
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

labels = train_df
X_test = pd.read_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/esm2.inference/{sys.argv[2]}/testing.fold.0.tokens.csv', index_col=0)
labels_test = pd.read_csv(f"{data['data_file_test']}", index_col=0)
# do for all scores
prediction=labels_test
y_columns = [col for col in labels.columns if col.startswith('score')]
for y_column in y_columns:
  y = labels[y_column]
  y = y.astype(float)

  X_train, y_train = X_new, y

  y_test = labels_test[y_column]
  y_test = y_test.astype(float)

  elastic_net_regressor = ElasticNetCV(cv=5, l1_ratio=np.linspace(0.01, 1, 21), tol=0.0001, max_iter=10000, n_jobs=64)

  elastic_net_regressor.fit(X_train, y_train)
    
  #prediction['is.train']=np.isin(prediction['genes'], X_train.index.values)
  prediction[f'prediction.{y_column}']=elastic_net_regressor.predict(X_test)
  # prediction['prediction_prob']=elastic_net_regressor.predict_proba(X_test)[:,0]
  prediction.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/elastic.net.result/{sys.argv[2]}/prediction.test.subset.{sys.argv[3]}.seed.{sys.argv[4]}.csv')
