import numpy as np
from scipy.stats import multivariate_normal, norm


def multi_gauss(test, train):
    return multivariate_normal.pdf(test, mean=np.mean(train, axis=1), cov=np.cov(train))


def norm_value(value, mean, std):
    return norm(mean, std).pdf(value)


def multi_gauss_test_mean_cov(test, mean, cov):
    return multivariate_normal.pdf(test, mean=mean, cov=cov)


def print_mean_std(x):
    m = np.mean(x, axis=0)
    s = np.std(x, ddof=1, axis=0)
    print("values: {}, mean: {}, std: {}".format(x, m, s))


if __name__ == "__main__":
    x = np.array([180, 160, 200, 171, 159, 150])
    print_mean_std(x)

    x = np.array([[-2, 2], [-1, 3], [0, 1], [-2, 1]])
    print_mean_std(x)
