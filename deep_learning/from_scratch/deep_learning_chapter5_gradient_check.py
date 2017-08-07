import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from deep_learning_chapter5 import TwoLayerNet

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    batch_size = 3
    x_batch = x_train[:batch_size]
    t_batch = t_train[:batch_size]
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ": " + str(diff))