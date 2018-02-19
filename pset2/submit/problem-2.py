"""
CS 6375
Prof. Nick Ruozzi
Problem Set 2
Question 2: Decision Tree
Author: Tri Minh Cao
Date: September 2017
"""

import numpy as np
from DecisionTree import DecisionTree
from collections import deque

def read_in(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            info = line.strip('\n').split(',')
            X.append([i for i in info[1:]])
            y.append(info[0])
    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    # read in the data
    X_train, y_train = read_in('./hw2_data/mush_train.data')
    X_test, y_test = read_in('./hw2_data/mush_test.data')

    clf = DecisionTree()
    clf.fit(X_train, y_train)

    print('Tree depth:', clf.depth)
    preds = clf.predict(X_test)
    train_preds = clf.predict(X_train)
    print('Accuracy on train set:', np.mean(train_preds==y_train))
    print('Accuracy on test set:', np.mean(preds==y_test))

    print()
    # use BFS to traverse the tree

    # queue = deque()
    # root = clf.tree
    # print('Split feature:', root.split_feature)
    # queue.append(root)
    # # print(current_branch.split_feature)
    # feature_dict = clf.feature_dict
    # while queue: # check if queue is not empty
    #     current_branch = queue.popleft()
    #     split_feat = current_branch.split_feature
    #     # print('Split feature:', split_feat)
    #     if split_feat is not None:
    #         feats = feature_dict[split_feat]
    #         print('Children features:', feats)
    #         for feat in feats:
    #             child = current_branch.children[feat]
    #             queue.append(child)
    #             print('feat:', feat, ';', 'split feature:', child.split_feature,
    #                     ';', 'predict:', child.predict)

    # test depth-1 decision trees
    print('Trying all decision trees with depth=1...')
    for i in range(X_train.shape[1]):
        clf = DecisionTree(depth=1)
        clf.fit(X_train, y_train, split_feat=i)
        preds = clf.predict(X_test)
        print('Feature:', i+1)
        print('Accuracy on test set:', np.mean(preds==y_test))
