import numpy as np
import sys
sys.path.insert(0, '/Users/trimcao/Dropbox/Richardson/Fall-2017/cs6375-ml-ruozzi/solution/lib')
sys.path.insert(0, '/home/trimcao/Dropbox/Richardson/Fall-2017/cs6375-ml-ruozzi/solution/lib')
from DecisionTree import DecisionTree


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
