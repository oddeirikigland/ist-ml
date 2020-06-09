import numpy as np
from sklearn.naive_bayes import GaussianNB
from normal_distribution import norm_value, multi_gauss

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


# print(multi_gauss(test, c0))
# print(multi_gauss(test, c1))
# print(
#     0.5
#     * multi_gauss(100, np.array([[170, 60, 50]]))
#     * multi_gauss(225, np.array([[160, 160, 150]]))
# )
def gauss_props(query, column, name):
    mean = np.mean(column, axis=1)
    std = np.std(column, ddof=1)
    gaus = multi_gauss(query, column)
    print("{}, mean: {}, std: {}, gaus: {}".format(name, mean, std, gaus))

# One dim
gauss_props(10, [[0,0,20,20]], "x_1_a")
gauss_props(10, [[10,20,10,20]], "x_2_a")
gauss_props(10, [[30,30,50,50]], "x_1_b")
gauss_props(10, [[30,40,30,40]], "x_2_b")

# two dim
gauss_props([10,10], [[0,0,20,20],[10,20,10,20]], "x_a")
gauss_props([10,10], [[30,30,50,50],[30,40,30,40]], "x_b")