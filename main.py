import numpy as np
# import matplotlib
# from matplotlib import pyplot as plt


def sigmoid_no_beta(z):
    return 1. / (1 + np.exp(-z))


def sigmoid(z):
    return sigmoid_no_beta(z)


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Net(object):
    def __init__(self, layers):
        self.weights = []
        self.activation = sigmoid
        self.activation_prime = sigmoid_prime
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)

        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=1000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            serror = 0
            for i in range(len(X)):
                # i = np.random.randint(X.shape[0])
                a = [X[i]]

                for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
                # output layer
                error = y[i] - a[-1]
                serror += error

                deltas = [error * self.activation_prime(a[-1])]

                # we need to begin at the second to last layer
                # (a layer before the output layer)
                for l in range(len(a) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

                # reverse
                # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
                deltas.reverse()

                # backpropagation
                # 1. Multiply its output delta and input activation
                #    to get the gradient of the weight.
                # 2. Subtract a ratio (percentage) of the gradient from the weight.
                for j in range(len(self.weights)):
                    layer = np.atleast_2d(a[j])
                    delta = np.atleast_2d(deltas[j])
                    self.weights[j] += learning_rate * layer.T.dot(delta)

            avg_error = serror / len(X)

            if k % 100 == 0:
                print('epochs:\t{}\t\tavg_loss:\t{}'.format(k, avg_error))

    def predict(self, x):
        a = np.concatenate((np.array([[1]]), np.array([x])), axis=1)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


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
    X = [[1, 0],
         [0, 1],
         [0, -1],
         [-1, 0]]
    X = np.asarray(X)
    t = [0, 1, 1, 0]
    F = Net([2, 2, 1])
    F.fit(X, t)

    print("0 0:\t", F.predict([0, 0]))
    print("1 0:\t", F.predict([1, 0]))
    print("0 1:\t", F.predict([0, 1]))
    print("1 1:\t", F.predict([1, 1]))


if __name__ == '__main__':
    main()
