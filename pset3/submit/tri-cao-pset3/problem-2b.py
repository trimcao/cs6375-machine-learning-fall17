"""
PSET 3
Problem 2b
Author: Tri M. Cao
Email: trimcao@gmail.com
Date: October 2017
"""
import numpy as np
import random
import pickle
from DecisionTree import Branch, DecisionTree, BoostedTree

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

# create 88 decision trees
def gen_tree(attribute, X, y, weights=None):
    """
    Generate trees with height = 1.
               a
             /   \
    """
    trees = []
    for l1 in [-1, 1]:
        for l2 in [-1, 1]:
            # create a decision tree
            tree = DecisionTree()
            tree.labels = set(y)
            root = Branch()
            tree.tree = root
            # split attribute 1
            root.split_feature = attribute
            # left branch of root
            left = Branch()
            left.predict = l1
            root.children[0] = left
            # right branch of root
            right = Branch()
            right.predict = l2
            root.children[1] = right
            # append tree to the list
            trees.append(tree)
    return trees

# compute r
def find_r(trees, alpha, x, y, k=-1):
    """
    Compute exponential loss for one sample for every learner except k
    (if k is not -1, i.e. k is not None).
    Note: y is a single label, not a vector of label.
    """
    num_trees = len(trees)
    sum_predict = 0
    for i in range(num_trees):
        if i != k:
            sum_predict += alpha[i]*trees[i].predict_single(x)
    return np.exp(-y*sum_predict)

# compute exponential loss
def exp_loss(trees, alpha, X, y):
    loss = 0
    for i in range(X.shape[0]):
        loss += find_r(trees, alpha, X[i], y[i])
    return loss

def predict_single(trees, alpha, x):
    """
    Predict a single sample using trees.
    """
    result = 0
    for i in range(len(trees)):
        result += alpha[i]*trees[i].predict_single(x)
    if result >= 0:
        return 1
    else:
        return -1

def update_alpha(t, trees, alpha, X, y):
    """
    Update alpha of tree t.
    """
    sum_correct = 0
    sum_incorrect = 0
    for i in range(X.shape[0]):
        r = find_r(trees, alpha, X[i], y[i], k=t)
        pred = trees[t].predict_single(X[i])
        if pred == y[i]:
            sum_correct += r
        else:
            sum_incorrect += r
    new_alpha = 0.5*np.log(sum_correct/sum_incorrect)
    return new_alpha

def fit(trees, X, y, epoch=50):
    """
    Method that trains a boosted tree using coordinate descent.
    """
    alpha = [0 for each in trees]
    num_trees = len(trees)
    # 1 epoch = 1 loop over all trees
    for e in range(epoch):
        # just iterate over the trees with no special selection.
        for t in range(num_trees):
            # update alpha of t
            alpha[t] = update_alpha(t, trees, alpha, X, y)
        # display exponential loss
        if e%20 == 0:
            print('Epoch:', e+1)
            print('Exponential loss =', exp_loss(trees, alpha, X, y))
    return alpha

def predict(trees, alpha, X):
    preds = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        preds[i] = predict_single(trees, alpha, X[i])
    return preds

def main():
    X, y = read_in('hw3_data/heart_train.data')
    X_test, y_test = read_in('hw3_data/heart_test.data')
    # build the trees
    num_feats = X.shape[1]
    trees = []
    for i in range(num_feats):
        trees.extend(gen_tree(i, X, y))
    epoch = 500
    # running coordinate descent
    print('Training using coordinate descent with epoch=' + str(epoch))
    print('Please be patient... It will take a while...\n')
    alpha = fit(trees, X, y, epoch=epoch)
    # predict on test set
    test_preds = predict(trees, alpha, X_test)
    print('\nAccuracy of coordinate descent:', np.mean(test_preds==y_test))
    # print alpha from coordinate descent
    print('\nPrinting alpha from coordinate descent:')
    for i in range(len(alpha)):
        if not np.isclose(alpha[i], 0):
            print('alpha', i, '=', alpha[i])
        else:
             print('alpha', i, '= 0')
    # adaBoost
    print('\n\nTraining with adaBoost with M=20')
    boost = BoostedTree(X, y, M=20)
    boost.fit_hw2(X, y, X_test, y_test)
    print('\nalpha from adaBoost:', str(boost.stages))

if __name__ == '__main__':
    main()
