import numpy as np
import matplotlib.pyplot as plt
from math import log
from closed_form_learning import closed_form_lin_reg


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

pred_log, weights_log, error_log = closed_form_lin_reg(
    log_x_bias, t, query=np.array([1, -0.3])
)
pred_quad, weights_quad, error_quad = closed_form_lin_reg(
    quad_x_bias, t, query=np.array([1, -0.3])
)


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


plot_task_6(weights_log, weights_quad)
