import numpy as np
import random
import sys
sys.path.insert(0, '/Users/trimcao/Dropbox/Richardson/Fall-2017/cs6375-ml-ruozzi/solution/lib')
# print(sys.path)
from SVM import SVMPrimal, SVMDual

def read_in(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            info = line.strip('\n').split(',')
            # label is the first column
            X.append([float(i) for i in info[1:]])
            y.append(float(info[0]))
    X = np.array(X)
    y = np.array(y)
    # change the output to *-1 and 1* instead of *0 and 1*
    #y[y == 0] = -1
    return X, y

def primal():
    # Primal SVM
    print('Part 1: Primal SVM\n')
    best_valid_acc = -1
    best_c = -1
    c_test = [10**i for i in range(9)]
    clf = SVMPrimal()
    for c in c_test:
        clf.fit(X_train, y_train, c=c)
        pred = clf.predict(X_valid)
        acc = clf.accuracy(pred, y_valid)
        print('c =', c, ';', 'accuracy:', acc)
        if acc > best_valid_acc:
            best_valid_acc = acc
            best_c = c
    print("Best C:", best_c)
    print("Best validation accuracy:", best_valid_acc)


def dual():
    # grid search for best c and best sigma
    c_test = [10**i for i in range(9)]
    sigma_test = [10**i for i in range(-1, 4)]

    c_test = [10**i for i in range(2)]
    sigma_test = [10**i for i in range(0, 2)]

    c_best = -1
    sigma_best = -1
    best_acc = -1
    for c in c_test:
        for sigma in sigma_test:
            clf = SVMDual()
            clf.fit(X_train, y_train, c=c, sigma=sigma)
            train_preds = clf.predict(X_train)
            train_acc = clf.accuracy(y_train, train_preds)
            valid_preds = clf.predict(X_valid)
            valid_acc = clf.accuracy(y_valid, valid_preds)
            print('c =', c, ';', 'sigma =', sigma)
            print('training set accuracy =', round(train_acc,4), ';',
                    'valid set accuracy =', round(valid_acc,4), '\n')
            if valid_acc > best_acc:
                best_acc = valid_acc
                c_best = c
                sigma_best = sigma
    print('best c:', c_best, ';', 'best sigma:', sigma_best)

    # predict the test set
    clf = SVMDual()
    clf.fit(X_train, y_train, c=c_best, sigma=sigma_best)
    preds = clf.predict(X_test)
    acc = clf.accuracy(y_test, preds)
    print('\nTEST SET')
    print('using c =', c_best, 'and', 'sigma =', sigma)
    print('accuracy on test set:', acc)

if __name__ == "__main__":
    # read in the data
    X_train, y_train = read_in('./hw2_data/wdbc_train.data')
    X_valid, y_valid = read_in('./hw2_data/wdbc_valid.data')
    X_test, y_test = read_in('./hw2_data/wdbc_test.data')
    # run code
    primal()
    dual()
