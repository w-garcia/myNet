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
        self.W = 2 * np.random.random((layer_size, n_in)) - 1
        #         self.deltas = np.zeros(layer_size, n_in + 1)
        self.activations = np.zeros(layer_size)
        self.learning_rate = 0.1

    def z(self, x):
        return np.dot(self.W, x)

    def a(self, x):
        res = sigmoid(self.z(x))
        return res

    def forward(self, x):
        # x_tilda = np.concatenate([[1.0], x])
        res = self.a(x)
        self.activations = res

        return res

    def error(self, delta):
        return np.dot(delta, self.W.T) * sigmoid_prime(self.activations)

    def update_with_gradient(self, delta):
        grad = np.dot(self.activations, delta)
        self.W += self.learning_rate * grad


class Net(object):
    def __init__(self, n_in):
        # b || w
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
        error = (y - t) * sigmoid_prime(self.layers[-1].activations)
        return error[0]

    def backwards(self, loss):
        # Calc deltas
        deltas = [loss * sigmoid_prime(self.layers[-1].activations)[0]]
        for l in reversed(self.layers[:-1]):
            deltas.append(l.error(deltas[-1]))

        deltas.reverse()

        # update with ggradients
        for i, l in enumerate(self.layers):
            l.update_with_gradient(deltas[i])

    def train_loop(self, X, t, epochs):
        # no batch
        for ep in range(epochs):
            loss = 0
            for x_i, t_i in zip(X, t):
            # i = np.random.randint(len(X))
            #     x_i = X[i]
            #     t_i = t[i]
                y_i = self.h(x_i)
                loss += self.loss(y_i, t_i)
                self.backwards(loss)

            avg_loss = loss / len(X)

            # if ep % 1000 == 0:
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
    F.train_loop(X, t, 1000)

    print("0 0:\t", F.h([0, 0]))
    print("1 0:\t", F.h([1, 0]))
    print("0 1:\t", F.h([0, 1]))
    print("1 1:\t", F.h([1, 1]))


if __name__ == '__main__':
    main()
