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
