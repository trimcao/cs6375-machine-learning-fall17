"""
CS 6375
Fall 2017
Prof. Nick Ruozzi
Problem Set 4 - Problem 1a
Author: Tri Minh Cao
Date: November 2017
"""

import numpy as np
from sklearn.svm import LinearSVC

def read_in(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            info = line.strip('\n').split(',')
            # label is the first column
            X.append([float(i) for i in info[:-1]])
            y.append(float(info[-1]))
    X = np.array(X)
    y = np.array(y)
    # change the output to *-1 and 1* instead of *0 and 1*
    y[y == 0] = -1
    return X, y

def normalize_data(X, X_train):
    X_norm = X - X_train.mean(axis=0)
    X_norm = X_norm / X_train.std(axis=0)
    return X_norm

def main():
    X_train, y_train = read_in('hw4_data/spam_train.data')
    X_test, y_test = read_in('hw4_data/spam_test.data')
    X_valid, y_valid = read_in('hw4_data/spam_validation.data')

    # construct the W matrix with zero mean
    W = normalize_data(X_train, X_train)
    W_valid = normalize_data(X_valid, X_train)
    W_test = normalize_data(X_test, X_train)
    # find the covariance matrix
    cov_mat = np.cov(W.T)
    # compute eigenvalues and eigenvectors.
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    # sort eigenvalues and eigenvectors
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    print('Top six eigenvalues:', eig_vals[:6])
    print('Training SVM classifiers using PCA')
    for k in range(1,6):
        for c in [1, 10, 100, 1000]:
            U = eig_vecs[:,:k]
            X_tran = W.dot(U)
            clf = LinearSVC(C=c, random_state=0)
            clf.fit(X_tran, y_train)
            X_valid_tran = W_valid.dot(U)
            preds = clf.predict(X_valid_tran)
            error = 1 - np.mean(preds==y_valid)
            print('For k = ' + str(k) + ' and c = ' + str(c) +
                     ', valid error is ' + str(error))
    # train the classifier with best k/c pair
    # best k/c pair is k=1, c=10
    # Note: accuracy changes randomly
    k = 1
    c = 1000
    U = eig_vecs[:,:k]
    X_tran = W.dot(U)
    clf = LinearSVC(C=c,random_state=0)
    clf.fit(X_tran, y_train)
    X_test_tran = W_test.dot(U)
    preds = clf.predict(X_test_tran)
    error = 1 - np.mean(preds==y_test)
    print('Using k = 1 and c = 1000, test error is ' + str(error))
    print()

    # compare with the SVM without PCA
    print('Training SVM without PCA...')
    best_error = 1
    best_c = -1
    for c in [1, 10, 100, 1000]:
        clf = LinearSVC(C=c, random_state=0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_valid)
        error = 1 - np.mean(preds==y_valid)
        if error < best_error:
            best_error = error
            best_c = c
        print('For c = ' + str(c) +
             ', valid error is ' + str(error))
    # use best_c
    clf = LinearSVC(C=best_c, random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    error = 1 - np.mean(preds==y_test)
    print('With c =', best_c, ', test set error is ' + str(error))


if __name__ == '__main__':
    main()
