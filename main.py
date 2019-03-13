import numpy as np
# import matplotlib
# from matplotlib import pyplot as plt


def sigmoid_no_beta(z):
    return 1. / (1 + np.exp(-z))


def sigmoid(z):
    return sigmoid_no_beta(z)


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Layer(object):
    def __init__(self, n_in, layer_size):
        self.W = np.random.rand(layer_size, n_in)
        #         self.deltas = np.zeros(layer_size, n_in + 1)
        self.activations = np.zeros(layer_size)
        self.learning_rate = 0.010

    def z(self, x):
        return np.dot(self.W, x)

    def a(self, x):
        self.activations = self.z(x)
        res = sigmoid(self.activations)
        return res

    def forward(self, x):
        # b || x
        # x_tilda = np.concatenate([[1.0], x])
        res = self.a(x)

        return res

    def backward(self, delta):
        delta = np.dot(self.W.T, delta) * self.grad()
        self.W += self.learning_rate * np.dot(self.W.T, delta)
        return delta

    def grad(self):
        return sigmoid_prime(self.activations)


class Net(object):
    def __init__(self, n_in):
        self.layers = [
            Layer(n_in + 1, 3),  # Hidden layer
            Layer(3, 1)  # Output, a logistic regression fn
        ]

    def h(self, x):
        prev_out = np.concatenate([[1.0], x])
        for l in self.layers:
            prev_out = l.forward(prev_out)

        return prev_out[0]

    def loss(self, y, t):
        return (y - t) ** 2

    def backwards(self, loss):
        for l in reversed(self.layers[:-1]):
            loss = l.backward(loss)

    def train_loop(self, X, t, epochs):
        # no batch
        for ep in range(epochs):
            loss = 0
            for x_i, t_i in zip(X, t):
                y_i = self.h(x_i)
                loss += self.loss(y_i, t_i)
                self.backwards(loss)

            avg_loss = loss / len(X)
            print("Loss epoch={}:\t{}".format(ep, avg_loss))


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

    # f = plt.figure()
    # ax = f.add_subplot(111)
    #
    # ax.scatter(X1[:, 0], X1[:, 1], color='orange')
    # ax.scatter(X2[:, 0], X2[:, 1], color='green')
    #
    # plt.show()

    F = Net(2)
    F.train_loop(X, t, 30)



if __name__ == '__main__':
    main()
