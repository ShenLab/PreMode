# Load the LogisticRegression classifier
# Note, use CV for cross-validation as requested in the question
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Load some other sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
# Import other libraries
import pandas as pd, numpy as np
import os
import yaml
from matplotlib import pyplot as plt

# $1 is the dataset name
# $2 is the yaml directory to compare with
# percent_pc = 0.95
# load dataset
# X = pd.read_csv(f'/share/terra/Users/gz2294/ld1/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/{sys.argv[1]}/training.csv', index_col=0)
X = pd.read_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/figs/{sys.argv[1]}.csv', index_col=0)
X = X.dropna(subset=["pLDDT", "FoldXddG", "energy", "conservation"])
X = X[X['score'].values != 0]
os.makedirs(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/', exist_ok=True)
# get train & validation in deep learning training
# with open(f'{sys.argv[2]}/{sys.argv[1]}/{sys.argv[1]}.seed.0.yaml', 'r') as f:
#     data = yaml.safe_load(f)
# train_idx = np.load(f"{data['log_dir']}/splits.0.npz")['idx_train']
# train_df = pd.read_csv(f"{data['data_file_train']}", index_col=0).iloc[train_idx]

# X = X.iloc[train_idx]
# remove the score-0 samples
# X = X[train_df['score'].values != 0]
# do SVD on X before training
# Xu, Xs, Xvt = np.linalg.svd(X)
# def calc_k(S, percentge):
#   '''
#   identify the minimum k value to reconstruct
#   '''
#   k = 0
#   total = sum(np.square(S))
#   svss = 0 #singular values square sum
#   for i in range(np.shape(S)[0]):
#       svss += np.square(S[i])
#       if (svss/total) >= percentge:
#           k = i+1
#           break
#   return k

# k = calc_k(Xs, percent_pc) # get the number of k to reconstruct 0.9 square sum

# def buildSD(S, k):
#   '''
#   reconstruct k singular value diag matrix
#   '''
#   SD = np.eye(k) * S[:k]
#   return SD

# def buildSD_1(S, k):
#   '''
#   reconstruct k singular value diag matrix
#   '''
#   SD = np.eye(k) * 1/S[:k]
#   return SD


# Xu_new = pd.DataFrame(Xu[:len(Xu), :k], index=X.index)
# Xvt_new = Xvt[:k, :len(Xvt)]
# Xs_new = buildSD(Xs, k)
# Xs_new_1 = buildSD_1(Xs, k)
X_new = X[["pLDDT", "FoldXddG", "energy", "conservation"]]

y = X['score']
y = y.astype(int)

# X_train, y_train = , 
prediction=X
for seed in range(5):
  X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, stratify=y, random_state=seed)

  # X_test = pd.read_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/esm2.inference/{sys.argv[1]}/testing.tokens.csv', index_col=0)
  # labels_test = pd.read_csv(f"{data['data_file_test']}", index_col=0)
  # # remove the score-0 samples
  # X_test = X_test[labels_test['score'] != 0]
  # labels_test = labels_test.iloc[labels_test['score'] != 0]
  # y_test = labels_test['score']
  # y_test = y_test.astype(int)
  kernel = 1.0 * RBF(1.0)
  gp_classifier = GaussianProcessClassifier(kernel=kernel, random_state=0, max_iter_predict=1000).fit(X_train, y_train)
  # gp_classifier.fit(X_train, y_train)

  import pickle
  with open(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/gp.{seed}.pkl', 'wb') as outputfile:
    pickle.dump(gp_classifier, outputfile)
  
  #prediction['is.train']=np.isin(prediction['genes'], X_train.index.values)
  prediction[f'prediction_split_{seed}']=gp_classifier.predict(X_new)
  prediction[f'prediction_prob_split_{seed}']=gp_classifier.predict_proba(X_new)[:,0]
  
  # draw fpr tpr
  fpr, tpr, thresholds = metrics.roc_curve(y_test.astype(float),
                                          gp_classifier.predict_proba(X_test)[:,1],
                                          pos_label=1)
  plt.plot(fpr, tpr, label="AUC={:.3f}".format(metrics.auc(fpr, tpr)))
  plt.xticks(np.arange(0.0, 1.1, step=0.1))
  plt.xlabel("False Positive Rate", fontsize=15)
  plt.yticks(np.arange(0.0, 1.1, step=0.1))
  plt.ylabel("True Positive Rate", fontsize=15)
  plt.legend(prop={'size': 13}, loc='lower right')
  plt.show()
  plt.savefig(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/auc.test.{seed}.pdf')
  plt.close()

  with open(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/GP.report.{seed}.txt', 'w') as outputfile:
    outputfile.write('Training Summary:\n')
    outputfile.write(classification_report(y_train, gp_classifier.predict(X_train)))
    outputfile.write('\nTesting Summary:\n')
    outputfile.write(classification_report(y_test, gp_classifier.predict(X_test)))
    outputfile.write('\n')
    outputfile.write("Training accuracy: {}\n\n".format(gp_classifier.score(X_train, y_train)))
    outputfile.write("Testing accuracy: {}\n\n".format(gp_classifier.score(X_test, y_test)))
    outputfile.write("Testing AUC: {}".format(metrics.auc(fpr, tpr)))

prediction.to_csv(f'/share/terra/Users/gz2294/PreMode.final/analysis/gp.result/{sys.argv[1]}/prediction.csv')
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


