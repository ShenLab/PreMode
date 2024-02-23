import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels
import sys
import pandas as pd

class SupervisedPCA:

    def __init__(self, n_components=None, kernel_on_labels=None):
        self.n_components = n_components
        self.U = None
        self.mean_of_X = None
        if kernel_on_labels != None:
            self.kernel_on_labels = kernel_on_labels
        else:
            self.kernel_on_labels = "linear"

    def fit_transform(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.fit(X, Y)
        X_transformed = self.transform(X, Y)
        return X_transformed

    def delta_kernel(self, Y):
        Y = Y.ravel()
        n_samples = len(Y)
        delta_kernel = np.zeros((n_samples, n_samples))
        for sample_index_1 in range(n_samples):
            for sample_index_2 in range(n_samples):
                if Y[sample_index_1] == Y[sample_index_2]:
                    delta_kernel[sample_index_1, sample_index_2] = 1
                else:
                    delta_kernel[sample_index_1, sample_index_2] = 0
        return delta_kernel

    def fit(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        n = X.shape[1]
        H = np.eye(n) - ((1/n) * np.ones((n,n)))
        # B = pairwise_kernels(X=Y.T, Y=Y.T, metric=self.kernel_on_labels)
        B = self.delta_kernel(Y=Y)
        eig_val, eig_vec = LA.eigh( X.dot(H).dot(B).dot(H).dot(X.T) )
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            self.U = eig_vec[:, :self.n_components]
        else:
            self.U = eig_vec

    def transform(self, X, Y=None):
        # X_centered = self.center_the_matrix(the_matrix=X, mode="remove_mean_of_columns_from_columns")
        # X_transformed = (self.U.T).dot(X_centered)
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def transform_outOfSample_all_together(self, X):
        # X: rows are features and columns are samples
        # X = X - self.mean_of_X
        x_transformed = (self.U.T).dot(X)
        return x_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        # X_centered = self.center_the_matrix(the_matrix=X, mode="remove_mean_of_columns_from_columns")
        # X_transformed = (U.T).dot(X_centered)
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        # X_reconstructed = X_reconstructed + self.mean_of_X
        return X_reconstructed

    def reconstruct_outOfSample_all_together(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        # x = x - self.mean_of_X
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        # X_reconstructed = X_reconstructed + self.mean_of_X
        return X_reconstructed

    def center_the_matrix(self, the_matrix, mode="double_center"):
        n_rows = the_matrix.shape[0]
        n_cols = the_matrix.shape[1]
        vector_one_left = np.ones((n_rows,1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - ((1/n_rows) * vector_one_left.dot(vector_one_left.T))
        H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))
        if mode == "double_center":
            the_matrix = H_left.dot(the_matrix).dot(H_right)
        elif mode == "remove_mean_of_rows_from_rows":
            the_matrix = H_left.dot(the_matrix)
        elif mode == "remove_mean_of_columns_from_columns":
            the_matrix = the_matrix.dot(H_right)
        return the_matrix

if __name__ == "__main__":
    # $1 is the train data file name
    # $2 is the train label file name
    # $3 is the test data file name
    # $4 is the test data file name
    # $5 is the output train PCA file
    # $6 is the output test PCA file

    X_train = pd.read_csv(f'{sys.argv[1]}', index_col=0).T.to_numpy()
    y_train = pd.read_csv(f'{sys.argv[2]}', index_col=0)
    y_train = y_train[['score']].T.to_numpy()
    y_train = y_train.astype(int)
    X_test = pd.read_csv(f'{sys.argv[3]}', index_col=0).T.to_numpy()
    y_test = pd.read_csv(f'{sys.argv[4]}', index_col=0)
    y_test = y_test[['score']].T.to_numpy()
    y_test = y_test.astype(int)
    
    spca = SupervisedPCA(n_components=5)
    X_train_transformed = spca.fit_transform(X_train, y_train)
    X_test_transformed = spca.transform(X_test)
    np.save(f'{sys.argv[5]}', X_train_transformed)
    np.save(f'{sys.argv[6]}', X_test_transformed)

