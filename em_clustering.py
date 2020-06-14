import numpy as np
import matplotlib.pyplot as plt
from normal_distribution import norm_value, multi_gauss_test_mean_cov

np.set_printoptions(precision=4)


def e_step(points, priors, likelihoods, binary):
    norm_posteriors = []
    point_count = 0
    for point in points:
        print("For point: {}".format(point))
        joint_probs = []
        cluster_count = 1
        for prior, likelihood in zip(priors, likelihoods):
            print("\tFor cluster: {}".format(cluster_count))
            print("\t\tPrior: {}".format(prior))
            if binary:
                prob = np.prod(
                    [
                        probability if binary else 1 - probability
                        for binary, probability in zip(point, likelihood)
                    ]
                )
            else:
                prob = likelihood[point_count]
            print("\t\tLikelihood: {}".format(prob))
            joint_prob = prob * prior
            print("\t\tJoint Probability: {}".format(joint_prob))
            joint_probs.append(joint_prob)

            print()
            cluster_count += 1
        norm_posterior = joint_probs / np.linalg.norm(joint_probs)
        print("\tnorm_posterior: {}".format(norm_posterior))
        norm_posteriors.append(norm_posterior)
        point_count += 1
        print()
    return norm_posteriors


def m_step(training_data, norm_posteriors, binary):
    likelihoods = []
    priors = []
    norm_posteriors_trans = np.transpose(norm_posteriors)
    training_data_trans = np.transpose(training_data)
    for i in range(len(norm_posteriors_trans)):
        print("For cluster: {}".format(i + 1))
        posterior = norm_posteriors_trans[i]
        likelihood = [
            sum(data * posterior) / sum(posterior) for data in training_data_trans
        ]
        if not binary:
            means = likelihood
            stds = [
                np.sqrt(sum((data - mean) ** 2 * posterior) / sum(posterior))
                for data, mean in zip(training_data_trans, means)
            ]
            likelihood = []
            for mean, std in zip(means, stds):
                likelihood += [
                    norm_value(value[0], mean=mean, std=std) for value in training_data
                ]
        prior = sum(posterior) / sum(sum(norm_posteriors_trans))
        print("\tLikelihoods: {}".format(likelihood))
        print("\tPrior: {}".format(prior))
        likelihoods.append(likelihood)
        priors.append(prior)
    return likelihoods, priors


def em_clustering(training_data, priors, likelihoods, binary=True):
    norm_posteriors = e_step(training_data, priors, likelihoods, binary)
    print("Norm posteriors: {}".format(norm_posteriors))
    likelihoods, priors = m_step(training_data, norm_posteriors, binary)
    return likelihoods, priors


def check_probability_data(training_data, priors, likelihoods):
    probabilities = []
    for point in training_data:
        likelihood_res = []
        for likelihood in likelihoods:
            likelihood_res.append(
                np.prod([prob if b else 1 - prob for prob, b in zip(likelihood, point)])
            )
        probability = sum([a * b for a, b in zip(likelihood_res, priors)])
        probabilities.append(probability)
    return np.prod(probabilities)


def task_1():
    training_data = [
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
    ]
    priors = [1 / 3] * 3
    likelihoods = [[0.8, 0.5, 0.1, 0.1], [0.1, 0.5, 0.4, 0.8], [0.1, 0.1, 0.9, 0.2]]

    probability1 = check_probability_data(training_data, priors, likelihoods)

    likelihoods, priors = em_clustering(training_data, priors, likelihoods)

    probability2 = check_probability_data(training_data, priors, likelihoods)

    print("Probability model 1: {}".format(probability1))
    print("Probability model 2: {}".format(probability2))


def plot_cluster():
    # mean and std collected from running task 2
    x_axis = np.arange(-10, 10, 0.001)
    plt.figure()
    plt.scatter([4, 0, 1], [0, 0, 0])
    plt.plot(
        x_axis, norm_value(x_axis, mean=0.48304673078791405, std=0.7689629116157932)
    )
    plt.plot(x_axis, norm_value(x_axis, mean=2.28487339524027, std=1.724824652583472))
    plt.xlim(-4, 10)
    plt.show()


def task_2():
    training_data = [[4], [0], [1]]
    priors = [0.5] * 2
    likelihoods = [
        [norm_value(value[0], mean=0, std=1) for value in training_data],
        [norm_value(value[0], mean=1, std=1) for value in training_data],
    ]

    probability = check_probability_data(training_data, priors, likelihoods)
    print("Probability model 1: {}".format(probability))

    likelihoods, priors = em_clustering(
        training_data, priors, likelihoods, binary=False
    )

    probability = check_probability_data(training_data, priors, likelihoods)
    print("Probability model 2: {}".format(probability))
    # plot_cluster()


def hw3():
    data = [[2, 4], [4, 2], [0, 0]]
    priors = [0.7, 0.3]
    likelihoods = [
        [
            multi_gauss_test_mean_cov(value, mean=[0, 4], cov=[[1, 0], [0, 1]])
            for value in data
        ],
        [
            multi_gauss_test_mean_cov(value, mean=[4, 0], cov=[[1, 0], [0, 1]])
            for value in data
        ],
    ]
    print(likelihoods)

    likelihoods, priors = em_clustering(data, priors, likelihoods, binary=False)
