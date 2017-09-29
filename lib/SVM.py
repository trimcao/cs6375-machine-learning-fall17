
import numpy as np
import cvxpy as cvx
import random

class SVMPrimal:
    """
    SVM Primal optimization problem.
    """
    def __init__(self):
        self.X = None
        self.y = None
        self.W = None # feature weights
        self.b = None # bias

    def fit(self, X, y, c=1):
        """
        Train a SVM classifier using data (using Primal problem).
        """
        self.X = X
        self.y = y
        # parameters
        D = X.shape[1] # number of features
        N = X.shape[0] # number of samples
        # instantiate variables in cvxpy
        W = cvx.Variable(D)
        b = cvx.Variable()
        # loss function of the primal SVM
        loss = (0.5*cvx.sum_squares(W) +
                c*cvx.sum_entries(cvx.pos(1 - cvx.mul_elemwise(y, X*W + b))))
        # need to minimize loss/N to avoid error
        prob = cvx.Problem(cvx.Minimize(loss/N))
        prob.solve()
        # save results
        self.W = W.value
        self.b = b.value
        return self.W, self.b

    def predict(self, X):
        """
        Predict new samples using found W and b.
        """
        if self.W is None or self.b is None:
            print('W and b have not been found yet.')
            return
        preds = np.dot(X, self.W) + self.b
        preds[preds >= 0] = 1
        preds[preds < 0] = -1
        return preds

    @classmethod
    def accuracy(cls, y_pred, y_truth):
        return np.mean(y_pred == y_truth)


class SVMDual:
    """
    SVM Dual optimization problem.
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.c = None
        self.sigma = None
        self.Delta = None # Lagrangian variables
        self.b = None # bias

    def fit(self, X, y, c=1, sigma=0.1):
        """
        Train a SVM classifier using Dual problem and Gaussian Kernel.
        """
        self.X_train = X
        self.y_train = y
        # parameters
        self.c = c
        self.sigma = sigma
        self.D = X.shape[1] # number of features
        self.N = X.shape[0] # number of samples
        # instantiate variables in cvxpy
        W = cvx.Variable(self.D)
        b = cvx.Variable()
        Delta = cvx.Variable(self.N) # Lagrangian multipliers
        # Set up the dual problem
        k_matrix = self.kernel_matrix(X, sigma)
        first_term = cvx.quad_form(cvx.mul_elemwise(y, Delta), k_matrix)
        second_term = cvx.sum_entries(Delta)
        loss = -0.5*first_term + second_term
        # add constraints
        constraints = [Delta >= 0, Delta <= c]
        dual_sum = cvx.sum_entries(cvx.mul_elemwise(y, Delta))
        constraints.append(dual_sum == 0)
        # instantiate the problem
        prob = cvx.Problem(cvx.Maximize(loss), constraints)
        prob.solve()
        # find bias term
        self.Delta = Delta.value
        self.find_bias(Delta, X, y)
        return self.Delta, self.b

    def predict(self, X):
        """
        Make prediction on new samples.
        """
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred = 0
            for j in range(self.N):
                pred += self.Delta[j]*self.y_train[j]*self.gaussian_kernel(X[i],self.X_train[j],self.sigma)
            # print(pred)
            pred += self.b
            # print(pred)
            if pred < 0 and (not np.isclose(pred, 0)):
                preds[i] = -1
            elif np.isclose(pred, 0):
                preds[i] = random.choice([1, -1])
            else:
                preds[i] = 1
        return preds

    @classmethod
    def gaussian_kernel(cls, x, z, sigma):
        """
        Compute the Gaussian kernel (between two vectors x and z).
        """
        return np.exp(-np.dot((x-z),(x-z)) / (2*sigma**2))

    def kernel_matrix(self, X, sigma):
        """
        Compute kernel matrix.
        """
        N = self.N
        k_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                k_matrix[i, j] = self.gaussian_kernel(X[i], X[j], sigma)
        return k_matrix

    def find_bias(self, Delta, X, y):
        bias = None
        N = self.N
        c = self.c
        sigma = self.sigma
        # np.isclose is used to avoid error when comparing number
        for i in range(N):
            if (not np.isclose(c, Delta.value[i])) and (not np.isclose(Delta.value[i], 0)):
                # print('support vector idx:', i)
                # print(Delta.value[i])
                result = 0
                for j in range(N):
                    result += Delta.value[j]*y[j]*self.gaussian_kernel(X[i],X[j],sigma)
                bias = y[i] - result
                # print('bias =', y[i] - result)
                break
        self.b = bias
        return bias

    @classmethod
    def accuracy(cls, y_pred, y_truth):
        return np.mean(y_pred == y_truth)
