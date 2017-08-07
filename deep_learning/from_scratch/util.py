import numpy as np

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

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()
    return grad

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)