import numpy as np
import matplotlib.pyplot as plt
from error_functions import mean_square_error
from math import log


def closed_form_lin_reg(X, y, query, ridge_regression=False, lambda_reg=2):
    X_t = np.transpose(X)
    inv_body = np.dot(X_t, X)
    if ridge_regression:
        inv_body += lambda_reg * np.identity(4)
    X_t_X_inv = np.linalg.inv(inv_body)
    weights = np.dot(np.dot(X_t_X_inv, X_t), y)
    error = mean_square_error(np.dot(X, weights), y)
    if len(query) > 0:
        pred = np.dot(weights, query)
    else:
        pred = 0
    print("pred: {}, error: {}, weights: {}".format(pred, error, weights))
    return pred, weights, error


# remember to add the bias as first element in the x data, as done below!
x = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 3], [1, 3, 3]])
t = np.array([1.4, 0.5, 2, 2.5])
pred, weights, error = closed_form_lin_reg(x, t, query=np.array([1, 2, 3]))

x = np.array([[1, -2], [1, -1], [1, 0], [1, 2]])
t = np.array([1, 0, 0, 0])
pred, weights, error = closed_form_lin_reg(x, t, query=np.array([1, -0.3]))

x = np.array(
    [
        [1.0, 0.8, 0.64, 0.512],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.2, 1.44, 1.728],
        [1.0, 1.4, 1.96, 2.744],
        [1.0, 1.6, 2.56, 4.096],
    ]
)
t = np.array([24, 20, 10, 13, 12])
pred, weights, error = closed_form_lin_reg(x, t, query=[])
pred, weights, error = closed_form_lin_reg(
    x, t, query=[], ridge_regression=True, lambda_reg=2
)
