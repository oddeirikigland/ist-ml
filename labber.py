from math import log2, log
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from activation_functions import sigmoid, sigmoid_gradient, gradient_cross_entropy
from error_functions import sum_squared_error
from normal_distribution import multi_gauss

np.set_printoptions(precision=5)

# Gradient descent learning


def update_rule(x, t, w, gradient_func, learning_rate=1):
    sum1 = 0
    for point, target in zip(x, t):
        sum1 += gradient_func(point, target, w)
    return w - learning_rate * sum1


def stochastic_update_rule(
    x, t, w, gradient_func, learning_rate=1, one_example_only=False
):
    for point, target in zip(x, t):
        w = w - learning_rate * gradient_func(point, target, w)
        if one_example_only:
            return w
    return w


x = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 3], [1, 3, 3]])
t = np.array([1, 1, 0, 0])


def gradient_task_1(x, t, w):
    sig_body = 2 * np.dot(w, x)
    return -2 * (t - sigmoid(sig_body)) * sigmoid_gradient(sig_body) * x


# Gradient descent one update
w = np.array([1, 1, 1])
w = update_rule(x, t, w, gradient_task_1)
# print(w)

# Stochastic gradient descent
w = np.array([1, 1, 1])
w = stochastic_update_rule(x, t, w, gradient_task_1, one_example_only=True)
# print(w)


# Gradient descent one update, task 2
w = np.array([1, 1, 1])
w = update_rule(x, t, w, gradient_cross_entropy)
# print(w)

# Stochastic gradient descent, task 2
w = np.array([1, 1, 1])
w = stochastic_update_rule(x, t, w, gradient_cross_entropy, one_example_only=True)
# print(w)


def gradient_task_3(x, t, w):
    x_w_dot = np.dot(x, w)
    exp_func = np.exp(np.square(x_w_dot))
    return -2 * (t - exp_func) * x * x_w_dot * exp_func


# Stochastic gradient descent, task 3
x = np.array([[1, 1, 1]])
t = np.array([0])
w = np.array([0, 1, 0])
w = stochastic_update_rule(
    x, t, w, gradient_task_3, learning_rate=2, one_example_only=True
)
# print(w)
