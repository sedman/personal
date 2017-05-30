#ml practice
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_files
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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

'''
Randomized index ranging from 0 to 59999 for each iteration.
This is split into train_fold and test_fold. 
'''
def cross_validation(sgd_clf, x_train, y_train):
    skfolds = StratifiedKFold(n_splits=5, random_state=42)
    for train_index, test_index in skfolds.split(x_train, y_train): #40000, 20000
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

def precision_and_recall_score(y_true, y_pred):
    print("precision core : " + str(precision_score(y_true, y_pred)))
    print("recall core : " + str(recall_score(y_true, y_pred)))

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()

def plot_precision_vs_recall(precisions, recalls):
    pass


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

print("downloading the data")
x, y = get_data_and_target(download_data())
print("data downloaded")

#show_digit_as_image(x, 59999)

x_train, x_test, y_train, y_test = spilit_data_set(x, y)

x_train, y_train = shuffle_data_set(x_train, y_train)

print(x_train)
print(y_train)

binary_y_train = (y_train == 9)  # True for all 5s, False for all other digits.
sgd_clf = train_using_binary_class(x_train, binary_y_train)
result = sgd_clf.predict([x[59999]])
print(result)

#cross_validation(sgd_clf, x_train, binary_y_train)

accuracy = cross_val_score(sgd_clf, x_train, binary_y_train, cv=3, scoring="accuracy")
print(accuracy)

never_9_clf = Never9Classifier()
accuracy = cross_val_score(never_9_clf, x_train, binary_y_train, cv=3, scoring="accuracy")
print(accuracy)

y_train_pred = cross_val_predict(sgd_clf, x_train, binary_y_train, cv=3)
print(len(y_train_pred))

result_matrix = confusion_matrix(binary_y_train, y_train_pred)
print(result_matrix)
print(confusion_matrix(binary_y_train, binary_y_train))

#맞는것과 틀린것을 얼마나 잘 판단하느냐?
print("The precision of the classifier : " + str(result_matrix[0][0]/sum(result_matrix[0])))
#맞는것을 얼마나 잘 판단하느냐?
print("recall : ", str(result_matrix[0][0]/(result_matrix[0][0] + result_matrix[1][0])))

precision_and_recall_score(binary_y_train, y_train_pred)
print("f1_score : " + str(f1_score(binary_y_train, y_train_pred)))

some_digit = x[36000]
print("some digit", some_digit.shape)
y_scores = sgd_clf.decision_function([some_digit])
threshold = 0
print("y_score :", y_scores)
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

threshold = 200000
print("y_score :", y_scores)
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, x_train, binary_y_train, cv=3, method="decision_function")
print("y_score generated by decision function", y_scores)

precisions, recalls, thresholds = precision_recall_curve(binary_y_train, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

precision_and_recall_score(binary_y_train, (y_scores > 70000))

fpr, tpr, thresholds = roc_curve(binary_y_train, y_scores)
plot_roc_curve(fpr, tpr)

print("roc auc score : ", roc_auc_score(binary_y_train, y_scores))

print("Random Forest Classifier")
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, x_train, binary_y_train, cv=3, method="predict_proba")
print(y_probas_forest)
y_scores_forest = y_probas_forest[:, 1]
print(y_scores_forest)
fpr_forest, tpr_forest, thresholds_forest = roc_curve(binary_y_train, y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()