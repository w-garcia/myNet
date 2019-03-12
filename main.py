import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def sigmoid_no_beta(z):
    return 1. / (1 + np.exp(-z))


def sigmoid(z):
    return sigmoid_no_beta(z)


class Layer(object):
    def __init__(self, n_in, layer_size):
        self.W = np.random.rand(layer_size, n_in)

    def z(self, x):
        return np.dot(self.W, x)


class Net(object):
    def __init__(self, n_in, n_hidden):
        self.layers = [
            Layer(n_in, 10)
        ]

    def h(self, x):
        pass


def main():
    n = 100
    X1 = [[1, 0],
          [-1, 0]]
    X2 = [[0, 1],
          [0, -1]]

    for _ in range((n - 4) / 2):
        s = np.array([np.random.uniform(1.01, 10), np.random.uniform(0, 0.999)])
        if np.random.uniform() > 0.5:
            s[0] = -s[0]
        if np.random.uniform() > 0.5:
            s[1] = -s[1]

        X1.append(s)

    for _ in range((n - 4) / 2):
        s = np.array([np.random.uniform(0, 0.999), np.random.uniform(1.01, 10)])
        if np.random.uniform() > 0.5:
            s[0] = -s[0]
        if np.random.uniform() > 0.5:
            s[1] = -s[1]
        X2.append(s)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    t = np.concatenate([[0] * len(X1), [1] * len(X2)])
    X = np.concatenate([X1, X2])
    assert len(X) == n, 'Length not 100'

    f = plt.figure()
    ax = f.add_subplot(111)

    ax.scatter(X1[:, 0], X1[:, 1], color='orange')
    ax.scatter(X2[:, 0], X2[:, 1], color='green')

    plt.show()






if __name__ == '__main__':
    main()
