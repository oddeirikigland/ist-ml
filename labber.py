from math import log2, log
import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# LAB 1
# information gain = E_start - E
# want highest value


def entropy(values):
    sum1 = 0
    for v in values:
        sum1 += v * log2(v)
    return -sum1


# print(entropy([0.5, 0.5]))
# print(entropy([1]))
# print(entropy([1 / 3, 2 / 3]))
# print(entropy([1 / 6, 1 / 6, 1 / 3, 1 / 3]))
# print(entropy([1 / 3, 1 / 3, 1 / 3]))


def gini(values):
    sum1 = 0
    for v in values:
        sum1 += v * v
    return 1 - sum1


# print(gini([0.5, 0.5]))

# print(np.cov(np.array([[-2, -1, 0, -2], [2, 3, 1, 1]])))  # one array for each row

# c0 = [[170, 60, 50,], [160, 160, 150]]
# print(np.mean(c0, axis=1))
# print(np.cov(c0))

# LAB 2
# The maximum likelihood gaussian is defined by the sample mean vector and
# the covariance matrix

# Bayes rule
# Naive bayes
def naive_bayes():
    x = np.array(
        [
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    y = np.array([1, 0, 0, 0, 1, 1, 1])
    print(GaussianNB().fit(x, y).predict(np.array([[1, 1, 1, 1, 1]])))


# naive_bayes()

# Multivariate gaussian and 1-d gauss

c0 = np.array([[170, 60, 50], [160, 160, 150]])
c1 = np.array([[80, 90, 70], [220, 200, 190]])
test = np.array([100, 225])


def multi_gauss(test, train):
    return multivariate_normal.pdf(test, mean=np.mean(train, axis=1), cov=np.cov(train))


# print(multi_gauss(test, c0))
# print(multi_gauss(test, c1))
# print(
#     0.5
#     * multi_gauss(100, np.array([[170, 60, 50]]))
#     * multi_gauss(225, np.array([[160, 160, 150]]))
# )

# Lab 3
def mean_square_error(pred, y):
    return np.mean(np.square(y - pred))


def closed_form_lin_reg(X, y, query):
    X_t = np.transpose(X)
    weights = np.dot(np.dot(np.linalg.inv(np.dot(X_t, X)), X_t), y)
    error = mean_square_error(np.dot(X, weights), y)
    pred = np.dot(weights, query)
    print("pred: {}, error: {}, weights: {}".format(pred, error, weights))
    return pred, weights, error


# x = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 3], [1, 3, 3]])
# t = np.array([1.4, 0.5, 2, 2.5])
# pred, weights, error = closed_form_lin_reg(x, t, query=np.array([1, 2, 3]))

# x = np.array([[1, -2], [1, -1], [1, 0], [1, 2]])
# t = np.array([1, 0, 0, 0])
# pred, weights, error = closed_form_lin_reg(x, t, query=np.array([1, -0.3]))


def plot_points(points, target):
    plt.figure()
    for x, c in zip(points, target):
        plt.scatter(x[0], x[1], color="b" if c == 1 else "r")
        print(x)
        print(c)
    plt.grid()
    plt.show()


def plot_reg_points(points, target):
    plt.scatter(points, target)
    plt.grid()


def transform_points(points, func):
    return np.array(list(map(lambda x: func(x), points)))


def quadratic_feature_transform(points):
    return transform_points(points, lambda x: x * x)


def logarithmic_feature_transform(points):
    return transform_points(points, log)


def add_bias(points):
    return transform_points(points, lambda x: [1, x])


# plot_points(
#     quadratic_feature_transform(np.array(
#         [
#             [-0.95, 0.62],
#             [0.63, 0.31],
#             [-0.12, -0.21],
#             [-0.24, -0.5],
#             [0.07, -0.42],
#             [0.03, 0.91],
#             [0.05, 0.09],
#             [-0.83, 0.22],
#         ]
#     )),
#     np.array([0, 0, 1, 0, 1, 0, 1, 0]),
# )

x = np.array([3, 4, 6, 10, 12])
log_x = logarithmic_feature_transform(x)
log_x_bias = add_bias(log_x)
quad_x = quadratic_feature_transform(x)
quad_x_bias = add_bias(quad_x)
t = np.array([1.5, 9.3, 23.4, 45.8, 60.1])
# plot_reg_points(x, t)

# pred_log, weights_log, error_log = closed_form_lin_reg(
#     log_x_bias, t, query=np.array([1, -0.3])
# )
# pred_quad, weights_quad, error_quad = closed_form_lin_reg(
#     quad_x_bias, t, query=np.array([1, -0.3])
# )


def plot_formula(points, func, label="fig"):
    plt.plot(points, func(points), label=label)


def plot_task_6(weights_log, weights_quad):
    values = np.arange(0.0, 14.0)
    plt.figure()
    plot_formula(
        values, lambda x: weights_log[0] + weights_log[1] * np.log(x), label="log",
    )
    plot_formula(
        values, lambda x: weights_quad[0] + weights_quad[1] * x * x, label="quadratic"
    )
    plot_reg_points(x, t)
    plt.xlim(0, 14)
    plt.ylim(0, 70)
    plt.legend()
    plt.show()


# plot_task_6(weights_log, weights_quad)

# Gradient descent learning


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


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
print(w)

# Stochastic gradient descent
w = np.array([1, 1, 1])
w = stochastic_update_rule(x, t, w, gradient_task_1, one_example_only=True)
print(w)


def gradient_cross_entropy(x, t, w):
    return -x * (t - sigmoid(np.dot(w, x)))


# Gradient descent one update, task 2
w = np.array([1, 1, 1])
w = update_rule(x, t, w, gradient_cross_entropy)
print(w)

# Stochastic gradient descent, task 2
w = np.array([1, 1, 1])
w = stochastic_update_rule(x, t, w, gradient_cross_entropy, one_example_only=True)
print(w)


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
print(w)
