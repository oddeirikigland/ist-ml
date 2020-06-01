import numpy as np


def mean_square_error(pred, y):
    return np.mean(np.square(y - pred))


def half_mean_square_error(pred, y):
    print(pred)
    return 0.5 * mean_square_error(pred, y)
