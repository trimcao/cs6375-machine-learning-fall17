"""
Naive Bayes
Author: Tri Minh Cao
Email: trimcao@gmail.com
Date: November 2017
"""
import numpy as np
from scipy.stats import dirichlet

class GaussianNB:
    """
    Gaussian Naive Bayes.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        self.labels = set(y) # set of all possible labels
        self.N = X.shape[0] # number of training samples
        self.param_dict = {}
        # build the parameters dictionary
        for label in self.labels:
            idx = (y == label)
            y_prob = np.sum(idx) / self.N # probability that y == label
            X_label = X[idx] # the training samples where y == label
            mean = X_label.mean(axis=0)
            var = X_label.var(axis=0)
            self.param_dict[label] = (y_prob, mean, var)

    @classmethod
    def gaussian_pdf(cls, x, mean, var):
        return (1/np.sqrt(2*np.pi*var)) * np.exp(-np.power(x-mean,2)/(2*var))

    def compute_prob(self, x, y):
        """
        Compute the conditional probability P(X=x|Y=y).
        y is the value or label of the current sample.
        """
        prob, mean, var = self.param_dict[y]
        for i in range(len(x)):
            prob *= self.gaussian_pdf(x[i], mean[i], var[i])
        return prob

    def predict_single(self, x):
        best_y = 0
        best_prob = -1
        for y in self.labels:
            prob = self.compute_prob(x, y)
            if prob > best_prob:
                best_prob = prob
                best_y = y
        return best_y

    def predict(self, X):
        preds = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            preds[i] = self.predict_single(X[i])
        return preds


class MultinomialNB:
    """
    Multinomial Naive Bayes.
    """
    def __init__(self):
        # Note: we use log probability
        self.prob_y = None
        self.prob_x_given_y = None

    def fit(self, X, y):
        """
        We assume that every feature and label are numerical values.
        Note: we use log probability
        """
        N = X.shape[0] # number of training samples
        self.labels = set(y)
        L = len(self.labels) # number of different labels
        self.features = range(X.shape[1])
        D = len(self.features) # number of different features
        # initialize the conditional probability matrix
        self.prob_x_given_y = np.zeros((L, D)) # y is row, x is column
        self.prob_y = np.zeros((L,))
        # compute P(Y)
        for i in range(L):
            self.prob_y[i] = np.log(np.sum(y==i) / N)
        # compute P(X|Y)
        # num_chars_per_label = np.zeros((L,))
        for j in range(L):
            # find indices of all samples that has y = j
            idx = (y == j)
            X_j = X[idx]
            # number of words x in samples y=j / total number of words in samples y=j
            self.prob_x_given_y[j] = np.log(X_j.sum(axis=0) / X_j.sum())


    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i], _ = self.predict_single(X[i])
        return preds

    def predict_single(self, x):
        # use argmax to find the label
        L = self.prob_x_given_y.shape[0]
        log_probs = np.zeros((L,))
        for j in range(L):
            log_probs[j] = (x * self.prob_x_given_y[j]).sum() + self.prob_y[j]
        # print(log_probs)
        return np.argmax(log_probs), log_probs

    def log_prob_x_y(self, x, y):
        y = np.int(y)
        log_prob = (x * self.prob_x_given_y[y]).sum() + self.prob_y[y]
        return log_prob

    def log_prob_x_y_dirichlet(self, x, y):
        y = np.int(y)
        prob_x_given_y = self.prob_x_given_y[y].rvs()[0]
        prob_y = self.prob_y.rvs()[0]
        log_prob = (x * prob_x_given_y).sum() + prob_y[y]
        return log_prob

    def predict_single_dirichlet(self, x):
        # use argmax to find the label
        L = len(self.prob_x_given_y)
        log_probs = np.zeros((L,))
        for j in range(L):
            prob_x_given_y = self.prob_x_given_y[j].rvs()[0]
            prob_y = self.prob_y.rvs()[0]
            log_probs[j] = (x * prob_x_given_y).sum() + prob_y[j]
        # print(log_probs)
        return np.argmax(log_probs), log_probs


class MixtureNB:
    """
    Mixture of Multinomial NB models.
    """
    def __init__(self, K=5, L=3, D=4):
        """
        k: number of NB models
        D: number of features
        L: number of different labels
        """
        self.K = K
        self.L = L
        self.D = D
        self.models = [MultinomialNB() for i in range(K)]
        for i in range(K):
            p_y = np.random.random(L)
            p_y = np.log(p_y / p_y.sum())
            self.models[i].prob_y = p_y
            p_x_given_y = np.random.random((L,D))
            for j in range(L):
                p_x_given_y[j] /= p_x_given_y[j].sum()
            self.models[i].prob_x_given_y = np.log(p_x_given_y)
        # initialize lambda z
        p_z = np.random.random(K)
        p_z = p_z / p_z.sum()
        self.p_z = np.log(p_z)

    def em(self, X_train, y_train):
        """
        EM algorithm for Mixture of Naive Bayes models.
        """
        N = X_train.shape[0] # number of training samples
        K = self.K
        L = self.L
        D = self.D
        # q_z = self.q_z
        p_z = self.p_z
        models = self.models

        # E-step, update Qi(z)
        q_z = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                # update q_z for sample i and NB model j
                q_z[i,j] = p_z[j] + models[j].log_prob_x_y(X_train[i],y_train[i])
            # normalize q_z[i], then take log
            q_z[i] = np.exp(q_z[i])
            q_z[i] = np.log(q_z[i] / q_z[i].sum())

        # M-step, update parameters in each model
        # k = NB model number
        # l = value of y
        # d = value of x
        # NOTE: keep q_z and p_z to be probability, not log probability
        # calculate the p_y as probabilty for each model, only calculate the log
        # at the end
        q_z_nonlog = np.exp(q_z)
        p_z = np.sum(q_z_nonlog, axis=0)
        for k in range(K):
            # build the mttrix q_z[k] * X_train
            X_q = np.tile(q_z_nonlog[:,k], (D,1)).T * X_train
            # q_z_nonlog for z = l
            q_k = q_z_nonlog[:,k]
            p_k = p_z[k]
            models[k].prob_y = np.zeros(L)
            models[k].prob_x_given_y = np.zeros((L,D))
            for l in range(L):
                # matrix q_z for the rows when y == l
                models[k].prob_y[l] = np.log(q_k[y_train==l].sum() / p_z[k])
                # update P(x|y) for each model
                X_l = X_q[y_train==l] # part of X_q that has y == l
                models[k].prob_x_given_y[l] = np.log(X_l.sum(axis=0) / X_l.sum())
        # divide p_z by N
        p_z = np.log(p_z / N)

        # update the parameters
        # self.q_z = q_z
        self.p_z = p_z
        self.models = models

    def predict_prob_log(self, X):
        p_z = self.p_z
        K = self.K
        probs = []
        for i in range(X.shape[0]):
            prob = None
            for k in range(K):
                _, predict_prob = self.models[k].predict_single(X[i])
                if prob is None:
                    prob = np.exp(predict_prob) * np.exp(p_z[k])
                else:
                    prob += np.exp(predict_prob) * np.exp(p_z[k])
                # print(prob)
            probs.append(prob)
        return np.log(probs)

    def predict(self, X):
        preds_prob = self.predict_prob_log(X)
        return np.argmax(preds_prob, axis=1)


class MixtureNBDirichlet:
    """
    Mixture of Multinomial NB models.
    With Dirichlet distribution.
    """
    def __init__(self, K=5, L=3, D=4):
        """
        k: number of NB models
        D: number of features
        L: number of different labels
        """
        self.K = K
        self.L = L
        self.D = D
        self.models = [MultinomialNB() for i in range(K)]
        for i in range(K):
            self.models[i].prob_y = dirichlet([2]*L)
            # create the dirichlet distribution individually so I have
            # separate scipy.stats.dirichlet objects (and I can change
            # them later).
            prob_x_given_y = []
            for j in range(L):
                prob_x_given_y.append(dirichlet([2]*D))
            self.models[i].prob_x_given_y = prob_x_given_y
        # initialize lambda z
        # p_z = np.random.random(K)
        # p_z = p_z / p_z.sum()
        # self.p_z = np.log(p_z)
        self.p_z = dirichlet([2]*K)

    def em(self, X_train, y_train):
        """
        EM algorithm for Mixture of Naive Bayes models.
        """
        N = X_train.shape[0] # number of training samples
        K = self.K
        L = self.L
        D = self.D
        # q_z = self.q_z
        p_z = np.log(self.p_z.rvs()[0])
        models = self.models

        # E-step, update Qi(z)
        q_z = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                # update q_z for sample i and NB model j
                q_z[i,j] = p_z[j] + models[j].log_prob_x_y_dirichlet(X_train[i],y_train[i])
            # normalize q_z[i], then take log
            q_z[i] = np.exp(q_z[i])
            q_z[i] = np.log(q_z[i] / q_z[i].sum())

        # M-step, update parameters in each model
        # k = NB model number
        # l = value of y
        # d = value of x

        # NOTE: keep q_z and p_z to be probability, not log probability
        # calculate the p_y as probabilty for each model, only calculate the log
        # at the end
        q_z_nonlog = np.exp(q_z)
        p_z = np.sum(q_z_nonlog, axis=0)
        # print(p_z)
        for k in range(K):
            # build the mttrix q_z[k] * X_train
            X_q = np.tile(q_z_nonlog[:,k], (D,1)).T * X_train
            # q_z_nonlog for z = l
            q_k = q_z_nonlog[:,k]
            p_k = p_z[k]
            for l in range(L):
                # matrix q_z for the rows when y == l
                # models[k].prob_y.alpha[l] = np.int(q_k[y_train==l].sum())
                models[k].prob_y.alpha[l] += np.int(q_k[y_train==l].sum())
                # update P(x|y) for each model
                X_l = X_q[y_train==l] # part of X_q that has y == l
                # models[k].prob_x_given_y[l].alpha = X_l.sum(axis=0).astype(np.int64)
                models[k].prob_x_given_y[l].alpha += X_l.sum(axis=0).astype(np.int64)
        # divide p_z by N
        # p_z = np.log(p_z / N)

        # update the parameters
        # self.q_z = q_z
        # print(self.p_z.alpha)
        self.p_z.alpha += p_z.astype(np.int64)
        # print(self.p_z.alpha)
        self.models = models

    def predict_prob_log(self, X):
        p_z = self.p_z.rvs()[0]
        K = self.K
        probs = []
        for i in range(X.shape[0]):
            prob = None
            for k in range(K):
                _, predict_prob = self.models[k].predict_single_dirichlet(X[i])
                if prob is None:
                    prob = np.exp(predict_prob) * np.exp(p_z[k])
                else:
                    prob += np.exp(predict_prob) * np.exp(p_z[k])
                # print(prob)
            probs.append(prob)
        return np.log(probs)

    def predict(self, X):
        preds_prob = self.predict_prob_log(X)
        return np.argmax(preds_prob, axis=1)
