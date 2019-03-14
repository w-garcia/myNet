
import numpy as np
import random
import matplotlib.pyplot as plt
import keras


# https://codereview.stackexchange.com/questions/207139/pretty-printing-of-the-numpy-ndarrays
def ndtotext(A, w=None, h=None):
    if A.ndim==1:
        if w == None :
            return str(A)
        else:
            s ='['+' '*(max(w[-1],len(str(A[0])))-len(str(A[0]))) +str(A[0])
            for i,AA in enumerate(A[1:]):
                s += ' '*(max(w[i],len(str(AA)))-len(str(AA))+1)+str(AA)
            s +='] '
    elif A.ndim==2:
        w1 = [max([len(str(s)) for s in A[:,i]])  for i in range(A.shape[1])]
        w0 = sum(w1)+len(w1)+1
        s= u'\u250c'+u'\u2500'*w0+u'\u2510' +'\n'
        for AA in A:
            s += ' ' + ndtotext(AA, w=w1) +'\n'
        s += u'\u2514'+u'\u2500'*w0+u'\u2518'
    elif A.ndim==3:
        h=A.shape[1]
        s1=u'\u250c' +'\n' + (u'\u2502'+'\n')*h + u'\u2514'+'\n'
        s2=u'\u2510' +'\n' + (u'\u2502'+'\n')*h + u'\u2518'+'\n'
        strings=[ndtotext(a)+'\n' for a in A]
        strings.append(s2)
        strings.insert(0,s1)
        s='\n'.join(''.join(pair) for pair in zip(*map(str.splitlines, strings)))
    return s


X = [[1, 0],
     [0, 1],
     [0, -1],
     [-1, 0]]

y = [0, 1, 1, 0]


class1_x1 = np.array(random.sample(range(2, 100), 48))
class1_x2 = np.array(np.random.rand(48))

random_x1c1 = np.array(random.sample(range(0, 48), 20))
random_x2c1 = np.array(random.sample(range(0, 48), 20))
class1_x1[random_x1c1] = class1_x1[random_x1c1 ] *-1
class1_x2[random_x2c1] = class1_x2[random_x2c1 ] *-1

class2_x1 = np.array(np.random.rand(48))
class2_x2 = np.array(random.sample(range(2, 100), 48))

random_x1c2 = np.array(random.sample(range(0, 48), 20))
random_x2c2 = np.array(random.sample(range(0, 48), 20))
class2_x1[random_x1c2] = class2_x1[random_x1c2 ] *-1
class2_x2[random_x2c2] = class2_x2[random_x2c2 ] *-1

for i in range(len(class1_x1)):
    X.append([class1_x1[i], class1_x2[i]])
    y.append(0)

for i in range(len(class1_x1)):
    X.append([class2_x1[i], class2_x2[i]])
    y.append(1)


X = np.array(X)
Y = np.array(y)
fig = plt.figure()
plt.scatter(X[: ,0], X[: ,1], c=Y)
plt.show()
fig.savefig("data_x.png")


def sigmoid_no_beta(z):
    return 1. / ( 1+ np.exp(-z))

X, Y = X.T, Y.reshape(1, Y.shape[0])



def layer_sizes(X, Y):

    n_x = X.shape[0] # size of input layer`
    n_h = 2
    n_y =Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    W1 = 2 * np.random.random((n_h, n_x)) - 1
    b1 = np.zeros(shape=(n_h, 1))

    W2 = 2 * np.random.random((n_h, n_h)) - 1
    b2 = np.zeros(shape=(n_h, 1))

    W3 = 2 * np.random.random((n_y, n_h)) - 1
    b3 = np.zeros(shape=(n_y, 1))

    print (W2.shape)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.matmul(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)
    Z3 = np.matmul(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3}

    return A3, cache


def compute_cost(A3, Y, parameters):
    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A3), Y) + np.multiply((1 - Y), np.log(1 - A3))
    cost = - np.sum(logprobs) / m

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17

    return cost


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x * 2))


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']

    Z1 = cache['Z1']
    Z2 = cache['Z2']
    Z3 = cache['Z3']

    # Backward propagation: calculate dW1, db1, dW2, db2.

    dZ3 = A3 - Y
    dW3 = (1. / m) * np.matmul(dZ3, A2.T)
    db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.matmul(W3.T, dZ3)
    dZ2 = dA2 * sigmoid(Z2) * (1 - sigmoid(Z2))
    dW2 = (1. / m) * np.matmul(dZ2, A1.T)
    db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1. / m) * np.matmul(dZ1, X.T)
    db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3}

    return grads


def update_parameters(parameters, grads, learning_rate=0.9):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    dW3 = grads['dW3']
    db3 = grads['db3']

    # Update rule for each parameter

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    # print (W2.shape)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    for i in range(0, num_iterations + 1):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5

    return predictions


# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h=2, num_iterations=5000, print_cost=True)

predictions = predict(parameters, X)
print ('Accuracy: %d' % float(
    (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

W1 = parameters['W1']
b1 = parameters['b1']
W2 = parameters['W2']
b2 = parameters['b2']
W3 = parameters['W3']
b3 = parameters['b3']

print("Weight_1")
print(ndtotext(W1))
print("")
print ("Bias_1")
print(ndtotext(b1))
print("")

print("Weight_2")
print(ndtotext(W2))
print("")
print ("Bias_2")
print(ndtotext(b2))
print("")

print("Weight_3")
print(ndtotext(W3))
print("")
print ("Bias_3")
print(ndtotext(b3))
print("")

print ("Learning Rate: 0.9")
print ("Epochs: 5000")


class KerasBaseNet(object):
    def __init__(self, fit_x, fit_y):
        self.model = None

        self._build(fit_x, fit_y)

    def _build(self, fit_x, fit_y):
        pass

    def compile(self, fit_x, fit_y):
        pass

    def evaluate(self, tst_x, tst_y):
        eval = self.model.evaluate(tst_x, tst_y)
        scores_dict = {}
        for i, m in enumerate(self.model.metrics_names):
            if m == 'loss':
                continue
            if m == 'categorical_accuracy':
                scores_dict[m] = eval[i] * 100
                continue
            scores_dict[m] = eval[i]
        return scores_dict


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()


class KerasSimpleNet(KerasBaseNet):
    def __init__(self, fit_x, fit_y):
        super(KerasSimpleNet, self).__init__(fit_x, fit_y)
        self._build(fit_x, fit_y)

    def _build(self, fit_x, fit_y):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(3, input_shape=fit_x[0].shape, activation=keras.activations.sigmoid))
        self.model.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
        self.model.add(keras.layers.Dense(1, activation='sigmoid', name='preds'))

    def compile(self, fit_x, fit_y):
        sgd = keras.optimizers.SGD(lr=0.9)
        self.model.compile(loss='binary_crossentropy', optimizer=sgd,
                           metrics=['accuracy'])

        self.model.fit(fit_x, fit_y, batch_size=10, epochs=100, callbacks=[history])


kn = KerasSimpleNet(X.T, y)
kn.compile(X.T, y)

print(kn.evaluate(X.T, y))

model = kn.model
fig = plt.figure()
plt.plot(range(1, 101), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
fig.savefig('{}.png'.format(len(model.layers)))

for i, layer in enumerate(model.layers):
    layer_weights = layer.get_weights()[0]
    print ("layer {} weight matrix ".format(i))
    print(ndtotext(np.asarray(layer_weights)))
    print("")

    layer_biases  = layer.get_weights()[1]
    print ("layer {} bias matrix ".format(i))
    print(ndtotext(np.asarray(layer_biases)))
    print("")
    print("")


