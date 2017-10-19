"""
Decision Tree
Author: Tri Minh Cao
Email: trimcao@gmail.com
Date: September 2017
"""

import numpy as np
from util import probability, conditional_probability
from util import entropy, conditional_entropy
import random

class Branch:
    """
    Branch class represents a branch in the decision tree.
    """
    def __init__(self):
        self.predict = None # is this branch predictable
        self.X = None # features
        self.y = None # labels
        self.weights = None
        self.split_feature = None
        self.possible_features = None
        self.children = {}
        self.level = None # depth in the tree


class DecisionTree:
    """
    """
    def __init__(self, depth=float('inf')):
        self.tree = Branch()
        self.tree.level = 0 # a tree with only the root has depth = 0
        self.labels = None # possible labels
        self.feature_dict = None # feature dict that shows options for each feature
        self.max_depth = depth
        self.depth = 0

    def fit(self, X, y, weights=None, split_feat=None):
        self.tree.X = X
        self.tree.y = y
        self.tree.weights = weights
        self.labels = set(y)
        self.tree.possible_features = [i for i in range(X.shape[1])]
        # get feature dict
        feats = {}
        num_feats = X.shape[1]
        for i in range(num_feats):
            feats[i] = set(X[:,i])
        self.feature_dict = feats
        # expand the tree recursively
        self.expand_tree(branch=None, parent=None, split_feat=split_feat)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.predict_single(X[i,:]))
        return np.array(predictions)

    def predict_single(self, x):
        """
        Predict using the decision tree using features in X
        (currently single sample).
        """
        prediction = None
        subtree = self.tree
        while prediction is None:
            prediction = subtree.predict
            # print(prediction)
            next_split = subtree.split_feature
            # print(next_split)
            if next_split is not None:
                subtree = subtree.children[x[next_split]]
        return prediction

    def predictable(self, branch, parent):
        """
        Check if a branch in the tree is predictable
        (not need to recurse on)
        """
        y = branch.y
        X = branch.X
        weights = branch.weights
        if len(set(y)) == 1: # branch is pure
            branch.predict = y[0]
            return True
        elif X.shape[0] == 0: # no sample
            # create a node here, and predict using parent's
            # majority vote
            y_parent = parent.y
            weights_parent = parent.weights
            branch.predict = self.majority_vote(y_parent, weights_parent)
            return True
        elif branch.level == self.max_depth or len(branch.possible_features) == 0:
            # the tree reaches max depth or no remaining feature
            # predict using majority vote
            branch.predict = self.majority_vote(y, weights)
            return True
        else:
            return False

    def expand_tree(self, branch=None, parent=None, split_feat=None):
        """
        Recursive function to expand a tree.
        """
        # if the current branch is the root
        if branch is None:
            branch = self.tree
        # find the current depth of the tree
        if branch.level > self.depth:
            self.depth = branch.level
        X_cur = branch.X
        y_cur = branch.y
        weights = branch.weights
        possible_features = branch.possible_features
        # base conditions
        if self.predictable(branch, parent):
            return
        # compute split feature
        if split_feat is None:
            split_feat = self.select_feature(y_cur, X_cur, possible_features,
                                                weights)
        branch.split_feature = split_feat
        feat_vals = self.feature_dict[split_feat]
        for each in feat_vals:
            child = Branch()
            # split data based on the feature
            child.X = X_cur[X_cur[:,split_feat]==each]
            child.y = y_cur[X_cur[:,split_feat]==each]
            if weights is not None:
                child.weights = weights[X_cur[:,split_feat]==each]
            # number of remaining features
            child.possible_features = list(branch.possible_features)
            child.possible_features.remove(split_feat)
            # add depth info for the child
            child.level = branch.level + 1
            # link the child to the current branch
            branch.children[each] = child
            # call the function on children (recursively)
            self.expand_tree(branch=child, parent=branch)
        return True

    @classmethod
    def select_feature(cls, y, X, possible_features, weights=None):
        """
        Select the best feature to split in the decision
        tree.
        """
        best_info_gain = -1
        split_feat = -1
        for feat in possible_features:
            info_gain = entropy(y, weights) - conditional_entropy(y, X[:,feat], weights)
            # print(feat, info_gain)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                split_feat = feat
        return split_feat

    def majority_vote(self, y, weights):
        """
        Output the label with majority vote.
        """
        best_choice = None
        most_votes = -1
        for choice in self.labels:
            if weights is not None:
                cur_vote = np.sum(weights[y==choice])
            else:
                cur_vote = np.sum(y==choice)
            if cur_vote > most_votes:
                most_votes = cur_vote
                best_choice = choice
        return best_choice



class BoostedTree:
    """
    Boosted Decision Tree using adaBoost algorithm.
    """
    def __init__(self, X, y, M=8, tree_depth=2):
        self.X = X
        self.y = y
        self.M = M # number of trees to be trained
        self.trees = []
        self.weight_list = []
        self.stages = [] # stage value for each tree
        self.errors = []
        self.tree_depth = tree_depth
        # get parameters
        self.labels = set(y)
        # get feature dict
        feats = {}
        num_feats = X.shape[1]
        for i in range(num_feats):
            feats[i] = set(X[:,i])
        self.feature_dict = feats

    def fit(self, X, y):
        # instantiate parameters
        N = X.shape[0] # number of training samples
        weights = np.array([1/N for i in range(N)])
        self.weight_list.append(weights)
        # start adaBoost loop
        for i in range(self.M):
            # print('Iteration:', i+1)
            # train a decision tree using the current weight
            current_weights = self.weight_list[i]
            # print(current_weights)
            current_tree = DecisionTree(depth=self.tree_depth)
            current_tree.fit(X, y, current_weights)
            preds = current_tree.predict(X)
            # update parameters
            error = self.compute_error(preds, y, current_weights)
            stage = 0.5*np.log((1-error)/error)
            self.errors.append(error)
            self.stages.append(stage)
            self.trees.append(current_tree)
            # get the new weights
            new_weights = self.update_weights(preds, y, current_weights,
                                              stage, error)
            self.weight_list.append(new_weights)

    @classmethod
    def compute_error(cls, preds, y, current_weights):
        error = 0
        for cur_pred, cur_y, cur_weight in zip(preds, y, current_weights):
            if cur_pred != cur_y:
                error += cur_weight
        return error

    @classmethod
    def update_weights(cls, preds, y, current_weights, stage, error):
        new_weights = []
        for cur_pred, cur_y, cur_weight in zip(preds, y, current_weights):
            weight = cur_weight * np.exp(-cur_y*cur_pred*stage)
            # normalize
            weight /= 2*np.sqrt(error*(1-error))
            new_weights.append(weight)
        return np.array(new_weights)

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(len(self.trees)):
            preds += self.trees[i].predict(X) * self.stages[i]
        # make discrete predictions
        preds[preds >= 0] = 1
        preds[preds < 0] = -1
        return preds

    def find_optimal_tree(self, X, y, weights):
        """
        Find optimal tree among all trees with 3 split attributes.
        """
        best_tree = None
        min_error = float('inf')
        num_feats = X.shape[1]
        for i in range(num_feats):
            for j in range(num_feats):
                for k in range(num_feats):
                    trees = []
                    trees.extend(self.gen_tree1([i, j, k], X, y))
                    trees.extend(self.gen_tree2([i, j, k], X, y))
                    trees.extend(self.gen_tree3([i, j, k], X, y))
                    trees.extend(self.gen_tree4([i, j, k], X, y))
                    trees.extend(self.gen_tree5([i, j, k], X, y))
                    # check weighted errors
                    for each_tree in trees:
                        preds = each_tree.predict(X)
                        error = self.compute_error(preds, y, weights)
                        if error < min_error:
                            min_error = error
                            best_tree = each_tree
        return best_tree, min_error


    def majority_vote(self, y, weights=None):
        """
        Output the label with majority vote.
        """
        best_choice = None
        most_votes = -1
        for choice in self.labels:
            if weights is not None:
                cur_vote = np.sum(weights[y==choice])
            else:
                cur_vote = np.sum(y==choice)
            if cur_vote > most_votes:
                most_votes = cur_vote
                best_choice = choice
        return best_choice

    def fit_hw(self, X, y, X_test, y_test):
        """
        Special fit function for the homework, problem 2a.
        """
        # enumerate all possible feature combinations
        # for each combo, create 5 trees
        # check the weighted error on each
        # find the tree with least weighted error
        # update the weights
        # instantiate parameters
        N = X.shape[0] # number of training samples
        weights = np.array([1/N for i in range(N)])
        self.weight_list.append(weights)
        # start adaBoost loop
        for i in range(self.M):
            print('Iteration:', i+1)
            # train a decision tree using the current weight
            current_weights = self.weight_list[i]
            # print(current_weights)
            current_tree, error = self.find_optimal_tree(X, y, current_weights)
            # current_tree.fit(X, y, current_weights)
            preds = current_tree.predict(X)
            # update parameters
            # error = self.compute_error(preds, y, current_weights)
            stage = 0.5*np.log((1-error)/error)
            self.errors.append(error)
            self.stages.append(stage)
            self.trees.append(current_tree)
            # get the new weights
            new_weights = self.update_weights(preds, y, current_weights,
                                              stage, error)
            self.weight_list.append(new_weights)
            # predict on training and test sets
            train_preds = self.predict(X)
            print('Accuracy on train set:', np.mean(train_preds==y))
            test_preds = self.predict(X_test)
            print('Accuracy on test set:', np.mean(test_preds==y_test))


    def gen_tree1(self, attributes, X, y, weights=None):
        """
        Brute-force method to solve problem 2, pset3, ML Fall 2017.
                   a
                 /   \
                b     c
        """
        trees = []
        for l1 in [-1, 1]:
            for l2 in [-1, 1]:
                for l3 in [-1, 1]:
                    for l4 in [-1, 1]:
                        # create a decision tree
                        tree = DecisionTree()
                        tree.labels = set(y)
                        root = Branch()
                        # root.X = X
                        # root.y = y
                        tree.tree = root
                        # split attribute 1
                        root.split_feature = attributes[0]
                        # left branch of root
                        left = Branch()
                        left.split_feature = attributes[1]
                        root.children[0] = left
                        # child 1
                        child1 = Branch()
                        child1.predict = l1
                        left.children[0] = child1
                        # child 2
                        child2 = Branch()
                        child2.predict = l2
                        left.children[1] = child2
                        # right branch of root
                        right = Branch()
                        right.split_feature = attributes[2]
                        root.children[1] = right
                        # child 3
                        child3 = Branch()
                        child3.predict = l3
                        right.children[0] = child3
                        # child 4
                        child4 = Branch()
                        child4.predict = l4
                        right.children[1] = child4
                        # append tree to the list
                        trees.append(tree)
        return trees

    def gen_tree2(self, attributes, X, y, weights=None):
        """
        Brute-force method to solve problem 2, pset3, ML Fall 2017.
                   a
                 /   \
                b
              /   \
             c
        """
        trees = []
        for l1 in [-1, 1]:
            for l2 in [-1, 1]:
                for l3 in [-1, 1]:
                    for l4 in [-1, 1]:
                        # create a decision tree
                        tree = DecisionTree()
                        tree.labels = set(y)
                        root = Branch()
                        # root.X = X
                        # root.y = y
                        tree.tree = root
                        # split attribute 1
                        root.split_feature = attributes[0]
                        # left branch of root
                        left = Branch()
                        left.split_feature = attributes[1]
                        root.children[0] = left
                        # right branch of root
                        right = Branch()
                        right.predict = l1
                        root.children[1] = right
                        # left child of left
                        left2 = Branch()
                        left2.split_feature = attributes[2]
                        left.children[0] = left2
                        # right child of left
                        right2 = Branch()
                        right2.predict = l2
                        left.children[1] = right2
                        # left3
                        left3 = Branch()
                        left3.predict = l3
                        left2.children[0] = left3
                        # right3
                        right3 = Branch()
                        right3.predict = l4
                        left2.children[1] = right3
                        # append tree to the list
                        trees.append(tree)
        return trees

    def gen_tree3(self, attributes, X, y, weights=None):
        """
        Brute-force method to solve problem 2, pset3, ML Fall 2017.
                   a
                 /   \
                b
              /   \
                   c
        """
        trees = []
        for l1 in [-1, 1]:
            for l2 in [-1, 1]:
                for l3 in [-1, 1]:
                    for l4 in [-1, 1]:
                        # create a decision tree
                        tree = DecisionTree()
                        tree.labels = set(y)
                        root = Branch()
                        # root.X = X
                        # root.y = y
                        tree.tree = root
                        # split attribute 1
                        root.split_feature = attributes[0]
                        # left branch of root
                        left = Branch()
                        left.split_feature = attributes[1]
                        root.children[0] = left
                        # right branch of root
                        right = Branch()
                        right.predict = l1
                        root.children[1] = right
                        # left child of left
                        left2 = Branch()
                        left2.predict = l2
                        left.children[0] = left2
                        # right child of left
                        right2 = Branch()
                        right2.split_feature = attributes[2]
                        left.children[1] = right2
                        # left3
                        left3 = Branch()
                        left3.predict = l3
                        right2.children[0] = left3
                        # right3
                        right3 = Branch()
                        right3.predict = l4
                        right2.children[1] = right3
                        # append tree to the list
                        trees.append(tree)
        return trees

    def gen_tree4(self, attributes, X, y, weights=None):
        """
        Brute-force method to solve problem 2, pset3, ML Fall 2017.
                   a
                 /   \
                      b
                     /  \
                         c
        """
        trees = []
        for l1 in [-1, 1]:
            for l2 in [-1, 1]:
                for l3 in [-1, 1]:
                    for l4 in [-1, 1]:
                        # create a decision tree
                        tree = DecisionTree()
                        tree.labels = set(y)
                        root = Branch()
                        # root.X = X
                        # root.y = y
                        tree.tree = root
                        # split attribute 1
                        root.split_feature = attributes[0]
                        # left branch of root
                        left = Branch()
                        left.predict = l1
                        root.children[0] = left
                        # right branch of root
                        right = Branch()
                        right.split_feature = attributes[1]
                        root.children[1] = right
                        # left child of right
                        left2 = Branch()
                        left2.predict = l2
                        right.children[0] = left2
                        # right child of right
                        right2 = Branch()
                        right2.split_feature = attributes[2]
                        right.children[1] = right2
                        # left3
                        left3 = Branch()
                        left3.predict = l3
                        right2.children[0] = left3
                        # right3
                        right3 = Branch()
                        right3.predict = l4
                        right2.children[1] = right3
                        # append tree to the list
                        trees.append(tree)
        return trees

    def gen_tree5(self, attributes, X, y, weights=None):
        """
        Brute-force method to solve problem 2, pset3, ML Fall 2017.
                   a
                 /   \
                      b
                     /  \
                    c
        """
        trees = []
        for l1 in [-1, 1]:
            for l2 in [-1, 1]:
                for l3 in [-1, 1]:
                    for l4 in [-1, 1]:
                        # create a decision tree
                        tree = DecisionTree()
                        tree.labels = set(y)
                        root = Branch()
                        # root.X = X
                        # root.y = y
                        tree.tree = root
                        # split attribute 1
                        root.split_feature = attributes[0]
                        # left branch of root
                        left = Branch()
                        left.predict = l1
                        root.children[0] = left
                        # right branch of root
                        right = Branch()
                        right.split_feature = attributes[1]
                        root.children[1] = right
                        # left child of right
                        left2 = Branch()
                        left2.split_feature = attributes[2]
                        right.children[0] = left2
                        # right child of right
                        right2 = Branch()
                        right2.predict = l2
                        right.children[1] = right2
                        # left3
                        left3 = Branch()
                        left3.predict = l3
                        left2.children[0] = left3
                        # right3
                        right3 = Branch()
                        right3.predict = l4
                        left2.children[1] = right3
                        # append tree to the list
                        trees.append(tree)
        return trees


    def fit_hw2(self, X, y, X_test, y_test):
        """
        Special fit function for the homework, problem 2b.
        """
        N = X.shape[0] # number of training samples
        weights = np.array([1/N for i in range(N)])
        self.weight_list.append(weights)
        # start adaBoost loop
        for i in range(self.M):
            print('Iteration:', i+1)
            # train a decision tree using the current weight
            current_weights = self.weight_list[i]
            # print(current_weights)
            current_tree, error = self.find_optimal_tree2(X, y, current_weights)
            # current_tree.fit(X, y, current_weights)
            preds = current_tree.predict(X)
            # update parameters
            # error = self.compute_error(preds, y, current_weights)
            stage = 0.5*np.log((1-error)/error)
            self.errors.append(error)
            self.stages.append(stage)
            self.trees.append(current_tree)
            # get the new weights
            new_weights = self.update_weights(preds, y, current_weights,
                                              stage, error)
            self.weight_list.append(new_weights)
            # predict on training and test sets
            train_preds = self.predict(X)
            print('Accuracy on train set:', np.mean(train_preds==y))
            test_preds = self.predict(X_test)
            print('Accuracy on test set:', np.mean(test_preds==y_test))

    def gen_tree(self, attribute, X, y, weights=None):
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

    def find_optimal_tree2(self, X, y, weights):
        """
        Find optimal tree among all trees with 3 split attributes.
        """
        best_tree = None
        min_error = float('inf')
        num_feats = X.shape[1]
        for i in range(num_feats):
            trees = []
            trees.extend(self.gen_tree(i, X, y))
            # check weighted errors
            for each_tree in trees:
                preds = each_tree.predict(X)
                error = self.compute_error(preds, y, weights)
                if error < min_error:
                    min_error = error
                    best_tree = each_tree
        return best_tree, min_error
