import numpy as np
from scipy.stats import multivariate_normal, norm


def multi_gauss(test, train):
    return multivariate_normal.pdf(test, mean=np.mean(train, axis=1), cov=np.cov(train))


def norm_value(value, mean, std):
    return norm(mean, std).pdf(value)
