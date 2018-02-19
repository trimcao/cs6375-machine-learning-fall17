"""
CS 6375
Fall 2017
Prof. Nick Ruozzi
Problem Set 4 - Problem 2
Author: Tri Minh Cao
Date: November 2017
"""

import numpy as np
import random
import sys
import NaiveBayes
from NaiveBayes import MultinomialNB, MixtureNB, MixtureNBDirichlet
from scipy.stats import dirichlet


def read_in(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            info = line.strip('\n').split(',')
            # label is the first column
            X.append(info[1])
            y.append(info[0])
    return X, y


def preprocess(X, y):
    """
    Convert text strings to a data matrix of numbers.
    """
    random.seed(0) # choose a seed so I can reproduce the results
    label2int = {label:i for i, label in enumerate(set(y))}
    int2label = {i:label for i, label in enumerate(set(y))}
    chars = {'A', 'G', 'T', 'C'}
    char2int = {char:i for i, char in enumerate(chars)}
    int2char = {i:char for i, char in enumerate(chars)}
    counts = {i:0 for i in int2char}
    # feature matrix
    data_matrix = np.zeros((len(X), len(chars)))
    for i in range(data_matrix.shape[0]):
        string = X[i].split()[0]
        for char in string:
            if char == 'D':
                char = random.choice(['A', 'G', 'T'])
            elif char == 'N':
                char = random.choice(['A', 'G', 'C', 'T'])
                # print(char)
            elif char == 'S':
                char = random.choice(['C', 'G'])
            elif char == 'R':
                char = random.choice(['A', 'G'])
            data_matrix[i, char2int[char]] += 1
    # label vector
    labels = np.zeros((len(y),))
    for i in range(labels.shape[0]):
        labels[i] = label2int[y[i]]
    return data_matrix, labels, label2int, int2label, char2int, int2char

def em_normal(X_train, y_train, X_test, y_test):
    # Experiment: Train 10 Mixture of NB models
    print('Training Mixture of NB...')
    num_iter = 200
    acc_train = 0
    acc_test = 0
    for m in range(10):
        print('start training model number', m+1)
        np.random.seed(m+10)
        clf = MixtureNB(K=5, L=3, D=4)
        for i in range(num_iter):
            clf.em(X_train, y_train)
        print('finish EM.')
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        acc_train += np.mean(train_preds==y_train)
        acc_test += np.mean(test_preds==y_test)
        print('Train accuracy:', np.mean(train_preds==y_train))
        print('Test accuracy:', np.mean(test_preds==y_test))
        print()
    print('Average accuracy on training set:', acc_train/10)
    print('Average accuracy on test set:', acc_test/10)

def em_dirichlet(X_train, y_train, X_test, y_test):
    print('Training Mixture of NB with Dirichlet distribution...')
    num_iter = 200
    acc_train = 0
    acc_test = 0
    for m in range(10):
        print('start training model number', m+1)
        np.random.seed(m+10)
        clf = MixtureNBDirichlet(K=5, L=3, D=4)
        for i in range(num_iter):
            clf.em(X_train, y_train)
        print('finish EM.')
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        acc_train += np.mean(train_preds==y_train)
        acc_test += np.mean(test_preds==y_test)
        print('Train accuracy:', np.mean(train_preds==y_train))
        print('Test accuracy:', np.mean(test_preds==y_test))
        print()
    print('Average accuracy on training set:', acc_train/10)
    print('Average accuracy on test set:', acc_test/10)

def main():
    X, y = read_in('hw4_data/bio.data')
    X, y, label2int, int2label, char2int, int2char = preprocess(X, y)
    # gather the data
    X_train = X[:2126]
    y_train = y[:2126]
    X_test = X[2126:]
    y_test = y[2126:]

    em_normal(X_train, y_train, X_test, y_test)
    print()
    em_dirichlet(X_train, y_train, X_test, y_test)



if __name__ == '__main__':
    main()
