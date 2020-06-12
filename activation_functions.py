import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


def gradient_cross_entropy(x, t, w):
    return -x * (t - sigmoid(np.dot(w, x)))


def hyberbolic_tangent(x):
    return np.tanh(x)


def gradient_hyperbolic_tangent(x):
    return 1 - hyberbolic_tangent(x) ** 2


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def gradient_softmax(x):
    return softmax(x) * (1 - softmax(x))


def sgn(x):
    # if x == 0:
    #     return 0
    if x < 0:
        return -1
    return 1


def step(x):
    return 1 if x >= 0 else 0


if __name__ == "__main__":
    z1 = np.array([6, 1, 6])
    x1 = hyberbolic_tangent(z1)
    print("x1: {}".format(x1))

    w = np.array([[1, 1, 1], [1, 1, 1]])
    b = np.array([1, 1])
    z2 = np.dot(w, x1) + b
    x2 = hyberbolic_tangent(z2)
    print("x2: {}".format(x2))

    w = np.array([[0, 0], [0, 0]])
    b = np.array([0, 0])
    z3 = np.dot(w, x2) + b
    print("z3: {}".format(z3))
    x3 = hyberbolic_tangent(z3)
    print("x3: {}".format(x3))
    x3_soft = softmax(z3)
    print("x3_soft: {}".format(x3_soft))

    gradient_z3 = gradient_hyperbolic_tangent(z3)
    print("gradientz3: {}".format(gradient_z3))

    gradient_z2 = gradient_hyperbolic_tangent(z2)
    print("gradientz2: {}".format(gradient_z2))

    gradient_z1 = gradient_hyperbolic_tangent(z1)
    print("gradientz1: {}".format(gradient_z1))
