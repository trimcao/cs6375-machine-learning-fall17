import numpy as np

# read in the data
X = []
y = []

with open('./perceptron.data', 'r') as f:
    for line in f:
        info = line.strip('\n').split(',')
        X.append([float(i) for i in info[:4]])
        y.append(float(info[4]))

# make the numpy arrays
X = np.vstack(X)
y = np.array(y)

# parameters
N = len(y)
learning_rate = 1.

def f(W, b, X):
    """
    function f as mentioned in the lecture
    """
    return W.dot(X) + b

def predict(W, b, X):
    """
    Perceptron prediction.
    X is an array of samples (not just one sample).
    """
    pred = X.dot(W) + b
    # change predictions to 1 and -1
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    return pred

def accuracy(y_pred, y_truth):
    """
    Compute accuracy.
    """
    return np.mean(y_pred == y_truth)

def p1_1():
    """
    Problem 1.1. Standard gradient descent with step size = 1
    """
    print('Problem 1.1. Standard gradient descent with step size = 1')
    W = np.zeros(4) # initial weights
    b = 0 # initial bias
    # (sub)gradient descent
    for epoch in range(10000):
        # print data
        # print('Iteration: ', epoch)
        # print(W)
        # print(b)
        # check accuracy
        pred = predict(W, b, X)
        acc = accuracy(pred, y)
        # print('accuracy:', acc)
        if (acc == 1.0):
            print('Perferct classifier achieved...')
            print('Iteration: ', epoch)
            print('Weights:', W)
            print('Bias:',b)
            print()
            break
        # update W and b using gradient descent
        for i in range(N):
            if -y[i]*f(W, b, X[i]) >= 0:
                W += learning_rate * y[i]*X[i]
                b += learning_rate * y[i]

def p1_2():
    """
    Problem 1.2. Stochastic gradient descent
    """
    print('Problem 1.2. Stochastic gradient descent')
    W = np.zeros(4) # initial weights
    b = 0 # initial bias
    # stochastic (sub)gradient descent
    for epoch in range(1000000):
        # print data
        # print('Iteration: ', epoch)
        # print(W)
        # print(b)
        # check accuracy
        pred = predict(W, b, X)
        acc = accuracy(pred, y)
        # print('accuracy:', acc)
        if (acc == 1.0):
            print('Perferct classifier achieved...')
            print('Iteration: ', epoch)
            print('Weights:', W)
            print('Bias:',b)
            print()
            break
        # update W and b using stochastic gradient descent
        i = epoch % N # the sample used in this epoch
        if -y[i]*f(W, b, X[i]) >= 0:
            W += learning_rate * y[i]*X[i]
            b += learning_rate * y[i]

def p1_3():
    """
    Problem 1.3. Test different step sizes.
    """
    print('Problem 1.3. Test different step sizes')
    # test for different step sizes
    step_sizes = [10., 1., 0.1, 0.01]
    for learning_rate in step_sizes:
        W = np.zeros(4) # initial weights
        b = 0 # initial bias
        print('Learning Rate:', learning_rate)
        for epoch in range(10000):
            # check accuracy
            pred = predict(W, b, X)
            acc = accuracy(pred, y)
            if (acc == 1.0):
                # print data
                print('Iteration: ', epoch)
                print('Weights:', W)
                print('Bias:',b)
                print(b)
                print('accuracy:', acc)
                print()
                break
            # train using gradient descent
            for i in range(N):
                if -y[i]*f(W, b, X[i]) >= 0:
                    W += learning_rate * y[i]*X[i]
                    b += learning_rate * y[i]

def main():
    p1_1()
    p1_2()
    p1_3()

if __name__ == '__main__':
    main()
