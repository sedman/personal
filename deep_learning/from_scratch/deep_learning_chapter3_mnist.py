from dataset.mnist import load_mnist
import numpy as np
import pickle

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def softmax(X):
    EXP_X = np.exp(X - np.max(X))
    return EXP_X / np.sum(EXP_X)

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, y_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']

    Y1 = sigmoid(np.dot(X,  W1) + B1)
    Y2 = sigmoid(np.dot(Y1, W2) + B2)

    OUT = softmax(np.dot(Y2, W3) + B3)
    return OUT

def test1():
    network = init_network()
    x_test, y_test =  get_data()

    accuracy_cnt = 0;
    print(len(x_test))
    for idx in range(len(x_test)):
    #for idx in range(1):
        predicted_y = predict(network, x_test[idx])
        #print(predicted_y)
        p = np.argmax(predicted_y)
        #print(p)
        if p == y_test[idx]:
            accuracy_cnt += 1

    print("Accuracy : " + str(float(accuracy_cnt) / len(x_test)))

def test2(batch_size = 100):
    x, y = get_data()
    network = init_network()
    accuracy_cnt = 0
    for idx in range(0, len(x), batch_size):
        x_batch = x[idx: idx+batch_size]
        y_batch = predict(network, x_batch)
        print(y_batch.shape)
        max_idx = np.argmax(y_batch, axis=1)
        print(max_idx.shape)
        accuracy_cnt += np.sum(max_idx == y[idx: idx + batch_size])
    print("Accuracy : " + str(float(accuracy_cnt) / len(x)))
