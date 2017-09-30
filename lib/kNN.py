"""
K-Nearest Neighbor
Author: Tri Minh Cao
Email: trimcao@gmail.com
September 2017
"""

import numpy as np
from queue import PriorityQueue

class kNN:
    """
    k-Nearest Neighbor
    """
    def __init__(self, X_train, y_train, k=5):
        """
        k: number of nearest neighbors considered
        """
        self.X_train, self.mean, self.std = self.normalize(X_train)
        self.y_train = y_train
        self.k = k
        self.labels = set(y_train)

    def fit(self, X, y):
        # probably no fitting here
        pass

    def predict(self, X):
        X_normed = (X-self.mean) / self.std
        preds = np.zeros(X_normed.shape[0])
        for i in range(X_normed.shape[0]):
            pred = self.predict_single(X_normed[i])
            preds[i] = pred
        return preds

    def predict_single(self, x):
        """
        Predict single sample
        """
        N = self.X_train.shape[0]
        # use a max-priority queue to store the minimum k neighbors
        queue = PriorityQueue(maxsize=self.k)
        # iterate through the training set
        for i in range(N):
            # distance must be made negative because we use max priority queue
            distance = -1 * self.euclidean_dist(x, self.X_train[i])
            if not queue.full():
                queue.put((distance, i)) # tuple (distance, index)
            else:
                current_max = queue.get()
                if np.abs(distance) < np.abs(current_max[0]):
                    queue.put((distance, i))
                else:
                    queue.put(current_max)
            # print(distance)
            # print(queue.queue)
        # majority vote from k nearest neighbors
        count = {label:0 for label in self.labels}
        while not queue.empty():
            next_neighbor = queue.get()
            label = self.y_train[next_neighbor[1]]
            count[label] += 1
        best_label = None
        most_votes = -1
        for label in self.labels:
            cur_vote = count[label]
            if cur_vote > most_votes:
                most_votes = cur_vote
                best_label = label
        return best_label


    @classmethod
    def normalize(cls, X):
        """
        Scale so X_normed will have 0 mean and 1 variance.
        """
        # compute sample mean and variance
        # each feature will be scaled to: (x - mean) / variance
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normed = (X-mean) / std
        return X_normed, mean, std

    @classmethod
    def euclidean_dist(cls, x, z):
        """
        Computer euclidean distance between two vectors.
        """
        return np.linalg.norm(x-z)
