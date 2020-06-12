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
    return gaus


# One dim
pa, pb = 0.5, 0.5
x1a = gauss_props(10, [[0, 0, 20, 20]], "x_1_a")
x2a = gauss_props(10, [[10, 20, 10, 20]], "x_2_a")
x1b = gauss_props(10, [[30, 30, 50, 50]], "x_1_b")
x2b = gauss_props(10, [[30, 40, 30, 40]], "x_2_b")
print(
    "\np(A|query) = p(a)*p(query|x1)*p(query|x2) = pa*x1a*x2a = {}".format(
        pa * x1a * x2a
    )
)
print(
    "p(B|query) = p(b)*p(query|x1)*p(query|x2) = pb*x1b*x2b = {}".format(pb * x1b * x2b)
)
print("Since p(A|query) > p(B|query) the query is classified as class A\n")

# two dim
xa = gauss_props([10, 10], [[0, 0, 20, 20], [10, 20, 10, 20]], "x_a")
xb = gauss_props([10, 10], [[30, 30, 50, 50], [30, 40, 30, 40]], "x_b")
print("\np(A|query) = p(a)*p(query|xa) = pa*xa = {}".format(pa * xa))
print("p(B|query) = p(b)*p(query|xb) = pb*xb = {}".format(pb * xb))
print("Since p(A|query) > p(B|query) the query is classified as class A\n")
