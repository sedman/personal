import numpy as np
from dataset.mnist import load_mnist
from deep_learning_chapter4_2Layer import TwoLayerNet

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.8

    net = TwoLayerNet()
    net.init_param(input_layer_size=784, hidden_layer_size=50, output_layer_size=10)
    loss_list = []
    train_acccuracy_list = []
    test_accuracy_list = []

    iter_per_epoch = max(train_size/batch_size, 1)

    for idx in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        X = x_train[batch_mask]
        T = t_train[batch_mask]

        PREDICTED = net.forward(X)
        cross_entropy = lambda w: net.loss(PREDICTED, T)

        gradient_W1 = net.numerical_gradient(cross_entropy, net.W1)
        gradient_W2 = net.numerical_gradient(cross_entropy, net.W2)

        net.W1 = net.W1 - learning_rate*gradient_W1
        net.W2 = net.W2 - learning_rate*gradient_W2

        loss = net.loss(PREDICTED, T)
        print(str(idx) + ": " + str(loss))
        loss_list.append(loss)

        if idx%iter_per_epoch == 0:
            train_accuracy = net.accuracy(x_train, t_train)
            test_accuracy = net.accuracy(x_test, t_test)
            train_acccuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            print("train accuracy, test accuracy | " + str(train_accuracy) + ", " + str(test_accuracy))


