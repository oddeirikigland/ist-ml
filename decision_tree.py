import numpy as np
from math import log2

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
print(entropy([0.2, 0.4, 0.2, 0.2]))
print(0.6 * entropy([2 / 3, 1 / 3]) + 0.4 * entropy([0.5, 0.5]))


def gini(values):
    sum1 = 0
    for v in values:
        sum1 += v * v
    return 1 - sum1


# print(gini([0.5, 0.5]))
