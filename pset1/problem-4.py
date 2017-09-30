"""
Problem 4 - SVM
Author: Tri M. Cao
Date: September 10, 2017
"""

import numpy as np
from cvxpy import *

# read in the data
X = []
y = []

with open('./mystery.data', 'r') as f:
    for line in f:
        info = line.strip('\n').split(',')
        X.append([float(i) for i in info[:-1]])
        y.append(float(info[-1]))

# convert the dataset to numpy arrays
X = np.vstack(X)
y = np.array(y)

# feature engineering
D = X.shape[1]
features = []
# quadratic features
for i in range(D):
    for j in range(i, D):
        features.append(X[:,i]*X[:,j])
# cubic features
for i in range(D):
    for j in range(i, D):
        for k in range(j, D):
            features.append(X[:,i]*X[:,j]*X[:,k])
# create a new dataset called X2
X_add = np.vstack(features)
X2 = np.column_stack((X, X_add.T))

# formulate the SVM optimization problem with cvxpy
D = X2.shape[1]
N = len(y)
# declare weights and bias
W = Variable(D)
b = Variable()
# declare the optimization constraints
loss = sum_squares(W)
constraints = []
for i in range(N):
    forward = y[i] * (W.T*X2[i] + b)
    constraints.append(forward >= 1)
objective = Minimize(loss)
prob = Problem(objective, constraints)
# solving the optimization problem
print('Start solving the optimization problem...')
result = prob.solve(solver=None, verbose=False)
print('Optimization problem solved!\n')
print('Resulting Weights and Bias:')
print('W =', W.value)
print()
print('b =', b.value)
print()

# print the margin
print('The optimal margin is:', 1/np.linalg.norm(W.value))
print()

# print out the support vectors
print('The following data points are the support vectors:')
num_wrong = 0
for i in range(1000):
    value = y[i]*(np.dot(W.value.T, X2[i]) + b.value)
    if value < 0:
        num_wrong += 1
    elif value < 1:
        print('index =', i, '; W.T*X+b =', value)
