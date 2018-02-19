"""
CS 6375
Fall 2017
Prof. Nick Ruozzi
Problem Set 4 - Problem 1b
Author: Tri Minh Cao
Date: November 2017
"""

import numpy as np
import NaiveBayes
from NaiveBayes import GaussianNB


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

def feature_select(D, k, s, eig_vals, eig_vecs):
    """
    Select s features using the top k eigenvectors.
    eig_vals and eig_vecs must be sorted.
    Return an array of features indices.

    D: number of features in the training set.
    """
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    top_vecs = eig_vecs[:k]
    feature_dist = np.sum(top_vecs**2, axis=0) / k
    # sample s features using np.random.choice
    features = np.random.choice(D, s, p=feature_dist)
    # get the unique feature indices
    features = np.unique(features)
    return features


def main():
    X_train, y_train = read_in('hw4_data/spam_train.data')
    X_test, y_test = read_in('hw4_data/spam_test.data')
    X_valid, y_valid = read_in('hw4_data/spam_validation.data')
    D = X_train.shape[1] # number of features
    # construct the W matrix with zero mean and unit variance
    W = normalize_data(X_train, X_train)
    W_test = normalize_data(X_test, X_train)

    # Train a Gaussian NB model
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Accuracy of the NB model:', np.mean(preds==y_test))
    print()

    # compute eigenvalues and eigenvectors
    cov_mat = np.cov(W.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    # Train Naive Bayes classifiers using PCA feature selection
    print('Training Naive Bayes classifiers using PCA feature selection...')
    for k in range(1,11):
        for s in range(1,21):
            error = 0
            for _ in range(100):
                features = feature_select(D, k,s,eig_vals,eig_vecs)
                clf = GaussianNB()
                clf.fit(W[:,features], y_train)
                preds = clf.predict(W_test[:,features])
                error += 1 - np.mean(preds==y_test)
            ave_error = error/100
            print('k = ' + str(k) + ', s = ' + str(s) + ': average error is ' +
                     str(ave_error))

if __name__ == '__main__':
    main()
