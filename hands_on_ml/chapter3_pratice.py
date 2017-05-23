#ml practice
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_files
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def download_data():
    return fetch_mldata('MNIST original')

def get_data_and_target(mnist):
    return mnist["data"], mnist["target"]

def show_digit_as_image(x, idx = 1):
    some_digit_image = x[idx].reshape(28, 28)

    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("on")
    plt.show()


def spilit_data_set(x, y):
    return x[:60000], x[60000:], y[:60000], y[60000:]

def shuffle_data_set(x_train, y_train):
    shuffle_index = np.random.permutation(60000)
    return x_train[shuffle_index], y_train[shuffle_index]

def cross_validation(sgd_clf, x_train, y_train):
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(x_train, y_train):
        clone_clf = clone(sgd_clf)
        x_train_folds = x_train[train_index]
        y_train_folds = y_train[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]

        clone_clf.fit(x_train_folds, y_train_folds)
        y_pred = clone_clf.predict(x_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))

def train_using_binary_class(x_train, binary_y_train):
    sgd_clf = SGDClassifier(random_state=42)
    print(binary_y_train)
    print(len(binary_y_train))
    sgd_clf.fit(x_train, binary_y_train)
    return sgd_clf

class Never9Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

x, y = get_data_and_target(download_data())

#show_digit_as_image(x, 59999)

x_train, x_test, y_train, y_test = spilit_data_set(x, y)

x_train, y_train = shuffle_data_set(x_train, y_train)

print(x_train)
print(y_train)

binary_y_train = (y_train == 9)  # True for all 5s, False for all other digits.
sgd_clf = train_using_binary_class(x_train, binary_y_train)
result = sgd_clf.predict([x[59999]])
print(result)

cross_validation(sgd_clf, x_train, binary_y_train)

accuracy = cross_val_score(sgd_clf, x_train, binary_y_train, cv=3, scoring="accuracy")
print(accuracy)

never_9_clf = Never9Classifier()
accuracy = cross_val_score(never_9_clf, x_train, binary_y_train, cv=3, scoring="accuracy")
print(accuracy)