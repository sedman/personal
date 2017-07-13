import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def mean_sequared_error(Y, T):
    return 0.5 * np.sum((Y - T)**2)

def cross_entropy_error(Y, T):#only for one hot encoding
    if Y.ndim == 1:
        Y = Y.reshape(1, Y.size)
        T = T.reshape(1, T.size)
    batch_size = Y.shape[0]
    return -np.sum(T*np.log(Y)) / batch_size

def cross_entropy_error2(Y, T):#not for one hot encoding
    if Y.ndim == 1:
        Y = Y.reshape(1, Y.size)
        T = T.reshape(1, T.size)
    batch_size = Y.shape[0]
    return -np.sum(np.log(Y[np.arange(batch_size), T])) / batch_size

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_train, y_train, x_test, y_test

def get_mini_batch(X, Y, batch_size):
    train_size = X.shape[0]
    batch_mask = np.random.choice(train_size, batch_size)
    return X[batch_mask], Y[batch_mask]

def test():
    _, _, x_test, y_test = get_data()
    x_batch, y_batch = get_mini_batch(x_test, y_test, 10)

    print(x_batch.shape)
    print(y_batch.shape)

    cross_entropy_error(y_batch, y_batch)

    print(np.arange(10))
    print(y_batch[np.arange(10), y_batch])

