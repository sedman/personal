import unittest
import numpy as np
from deep_learning_chapter3_mnist import softmax, sigmoid
from deep_learning_chapter4 import cross_entropy_error

class TwoLayerNet:
    def __init__(self):
        pass

    def init_param(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.W1 = 0.1 * np.random.rand(input_layer_size, hidden_layer_size)
        self.B1 = np.zeros(hidden_layer_size)
        self.W2 = 0.1 * np.random.rand(hidden_layer_size, output_layer_size)
        self.B2 = np.zeros(output_layer_size)

    def forward(self, X1):
        X2 = sigmoid(np.dot(X1, self.W1) + self.B1)
        ret = softmax(np.dot(X2, self.W2) + self.B2)
        return ret

    def loss(self, predicted, target):
        return cross_entropy_error(predicted, target)

    def numerical_gradient(self, f, x):
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

    def accuracy(self, Y, T):
        return np.sum(Y == T) / float(np.size(Y))


class TestTwoLayerNet(unittest.TestCase):
    def setUp(self):
        self.net = TwoLayerNet()

    def test_init_param(self):
        self.net.init_param(input_layer_size=2, hidden_layer_size=3, output_layer_size=2)
        self.assertEqual(self.net.W1.shape, (2, 3))
        self.assertEqual(self.net.W2.shape, (3, 2))

    def test_forward(self):
        self.net.init_param(input_layer_size=2, hidden_layer_size=3, output_layer_size=2)
        X = np.array([2.0, 3.0])
        self.assertEqual(self.net.forward(X).shape, (2, ))

    def test_loss(self):
        self.net.init_param(input_layer_size=2, hidden_layer_size=3, output_layer_size=2)
        X = np.array([2.0, 3.0])
        predicted = self.net.forward(X)
        T = np.zeros(predicted.shape)
        T[np.argmax(predicted)] = 1
        loss_true = self.net.loss(predicted, T)

        T = np.zeros(predicted.shape)
        T[np.argmin(predicted)] = 1
        loss_false  = self.net.loss(predicted, T)
        self.assertLess(loss_true, loss_false, "false loss must be greater than true")

    def test_gradient_descent(self):
        self.net.init_param(input_layer_size=2, hidden_layer_size=3, output_layer_size=2)
        X = np.array([2.0, 3.0])
        predicted = self.net.forward(X)
        T = np.zeros(predicted.shape)
        T[np.argmax(predicted)] = 1

        loss_f = lambda w: self.net.loss(predicted, T)
        self.assertEqual(self.net.numerical_gradient(loss_f, self.net.W1).shape, self.net.W1.shape)
        self.assertEqual(self.net.numerical_gradient(loss_f, self.net.W2).shape, self.net.W2.shape)

    def test_accuracy(self):
        self.net.init_param(input_layer_size=2, hidden_layer_size=3, output_layer_size=2)
        X = np.array([2.0, 3.0])
        T = Y = self.net.forward(X)
        self.assertEqual(self.net.accuracy(Y, T), 1)
        T = np.zeros(Y.shape)
        self.assertEqual(self.net.accuracy(Y, T), 0)




