"""
PSET 3
Problem 2a
Author: Tri M. Cao
Email: trimcao@gmail.com
Date: October 2017
"""

import numpy as np
from DecisionTree import DecisionTree, BoostedTree

def read_in(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            info = line.strip('\n').split(',')
            X.append([int(i) for i in info[1:]])
            y.append(int(info[0]))
    X = np.array(X)
    y = np.array(y)
    # make the labels -1 and 1
    y[y==0] = -1
    return X, y


def main():
    X, y = read_in('hw3_data/heart_train.data')
    X_test, y_test = read_in('hw3_data/heart_test.data')
    # run adaBoost with M=10
    M = 10
    print('Running adaBoost with M=' + str(M))
    print('Please be patient... It will take a while...')
    boost = BoostedTree(X, y, M=M)
    boost.fit_hw(X, y, X_test, y_test)
    # print errors and stages (alphas)
    print('\nErrors:', boost.errors)
    print('alpha:', boost.stages)

if __name__ == '__main__':
    main()
