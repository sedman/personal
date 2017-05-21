#ml practice
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_files
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def download_data():
    return fetch_mldata('MNIST original')

def get_data_and_target(mnist):
    return mnist["data"], mnist["target"]

def show_digit_as_image(x, idx = 1):
    some_digit = x[idx]
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("on")
    plt.show()

def spilit_data_set(x, y):
    return x[:60000], x[60000:], y[:60000], y[60000:]

def shuffle_data_set(x_train, y_train):
    shuffle_index = np.random.permutation(60000)
    return x_train[shuffle_index], y_train[shuffle_index]

x, y = get_data_and_target(download_data())

x_train, x_test, y_train, y_test = spilit_data_set(x, y)

x_train, y_train = shuffle_data_set(x_train, y_train)

print(x_train)
print(y_train)