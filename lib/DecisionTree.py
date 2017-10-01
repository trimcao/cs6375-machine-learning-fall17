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
    def __init__(self, M=8, tree_depth=2):
        self.M = M # number of trees to be trained
        self.trees = []
        self.weight_list = []
        self.stages = [] # stage value for each tree
        self.errors = []
        self.tree_depth = tree_depth

    def fit(self, X, y):
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
        for i in range(self.M):
            preds += self.trees[i].predict(X) * self.stages[i]
        # make discrete predictions
        preds[preds > 0] = 1
        preds[preds < 0] = -1
        return preds
