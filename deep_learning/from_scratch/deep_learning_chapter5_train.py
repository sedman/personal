import numpy as np
from dataset.mnist import load_mnist
from deep_learning_chapter5 import TwoLayerNet
import matplotlib.pyplot as plt
#from optimizer import SGD
#from optimizer import Momentum
#from optimizer import AdaGrad
from optimizer import Adam
#from optimizer import RMSprop

def draw_acc_graph(epoch_list, train_acc_list, test_acc_list):
    plt.plot(epoch_list, train_acc_list, label="train accuracy")
    plt.plot(epoch_list, test_acc_list, label="test accuracy", linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

def draw_loss_graph(train_loss_list):
    plt.plot(train_loss_list, label="train loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.show()

def gradient_check():
    (x_train, t_train), (_, _) = \
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

def train():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 20000
    train_size = x_train.shape[0]
    batch_size = 100

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_list = []

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        optimizer = Adam()
        network.train(x_batch, t_batch, optimizer)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            epoch_list.append(i/iter_per_epoch)
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train accuracy : " + str(train_acc) + ", test accuracy : " + str(test_acc))
    print("final loss : " + str(loss))
    draw_acc_graph(epoch_list, train_acc_list, test_acc_list)
    draw_loss_graph(train_loss_list)
