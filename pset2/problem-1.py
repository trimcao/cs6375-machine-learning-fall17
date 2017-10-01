"""
CS 6375
Prof. Nick Ruozzi
Problem Set 2
Question 1: SVM and k-nearest neighbor
Author: Tri Minh Cao
Date: September 2017
"""

import numpy as np
import random
import sys
sys.path.insert(0, '/Users/trimcao/Dropbox/Richardson/Fall-2017/cs6375-ml-ruozzi/solution/lib')
sys.path.insert(0, '/home/trimcao/Dropbox/Richardson/Fall-2017/cs6375-ml-ruozzi/solution/lib')
# print(sys.path)
from SVM import SVMPrimal, SVMDual
from kNN import kNN

def read_in(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            info = line.strip('\n').split(',')
            # label is the first column
            X.append([float(i) for i in info[1:]])
            y.append(float(info[0]))
    X = np.array(X)
    y = np.array(y)
    # change the output to *-1 and 1* instead of *0 and 1*
    #y[y == 0] = -1
    return X, y

def primal():
    # Primal SVM
    print('\nPart 1: Primal SVM\n')
    best_valid_acc = -1
    best_c = -1
    c_test = [10**i for i in range(9)]
    for c in c_test:
        clf = SVMPrimal()
        clf.fit(X_train, y_train, c=c)
        valid_preds = clf.predict(X_valid)
        train_preds = clf.predict(X_train)
        train_acc = clf.accuracy(train_preds, y_train)
        valid_acc = clf.accuracy(valid_preds, y_valid)
        print('c =', c)
        print('train set accuracy =', round(train_acc,4), ';',
                'valid set accuracy =', round(valid_acc,4))
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_c = c
    print("\nBest C:", best_c)
    print("Best validation accuracy:", best_valid_acc)
    # predict the test set
    clf = SVMPrimal()
    clf.fit(X_train, y_train, c=best_c)
    preds = clf.predict(X_test)
    acc = clf.accuracy(y_test, preds)
    print('\nTEST SET')
    print('using c =', best_c)
    print('accuracy on test set:', acc)

def dual():
    print('\nPart 2: SVM Dual')
    # grid search for best c and best sigma
    c_test = [10**i for i in range(9)]
    sigma_test = [10**i for i in range(-1, 4)]

    # c_test = [10**i for i in range(8,9)]
    # sigma_test = [10**i for i in range(-1, 4)]

    print('Computing kernel matrices...')
    k_matrices = {}
    for sigma in sigma_test:
        k_matrices[sigma] = SVMDual.kernel_matrix(X_train, sigma)

    c_best = -1
    sigma_best = -1
    best_acc = -1
    for c in c_test:
        for sigma in sigma_test:
            clf = SVMDual()
            print('c =', c, ';', 'sigma =', sigma)
            try:
                clf.fit(X_train, y_train, c=c, sigma=sigma, k_matrix=k_matrices[sigma])
                train_preds = clf.predict(X_train)
                train_acc = clf.accuracy(y_train, train_preds)
                valid_preds = clf.predict(X_valid)
                valid_acc = clf.accuracy(y_valid, valid_preds)
                print('train set accuracy =', round(train_acc,4), ';',
                        'valid set accuracy =', round(valid_acc,4), '\n')
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    c_best = c
                    sigma_best = sigma
            except:
                print('Solver cannot find solution.\n')
    print('best c:', c_best, ';', 'best sigma:', sigma_best)

    # predict the test set
    clf = SVMDual()
    clf.fit(X_train, y_train, c=c_best, sigma=sigma_best, k_matrix=k_matrices[sigma_best])
    preds = clf.predict(X_test)
    acc = clf.accuracy(y_test, preds)
    print('\nTEST SET')
    print('using c =', c_best, 'and', 'sigma =', sigma_best)
    print('accuracy on test set:', acc)


def kNearest():
    print('\nPart 3: k-Nearest Neighbor')
    k_test = [1, 5, 11, 15, 21]
    for k in k_test:
        clf = kNN(X_train, y_train, k=k)
        preds = clf.predict(X_test)
        print('k =', k, ';', 'accuracy on test set =', np.mean(preds==y_test))


if __name__ == "__main__":
    # read in the data
    X_train, y_train = read_in('./hw2_data/wdbc_train.data')
    X_valid, y_valid = read_in('./hw2_data/wdbc_valid.data')
    X_test, y_test = read_in('./hw2_data/wdbc_test.data')
    # run code
    # primal()
    # dual()
    kNearest()
