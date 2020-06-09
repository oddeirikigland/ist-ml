import numpy as np


def get_mean_vector(data):
    return np.mean(data, axis=0)


def get_covariance_matrix(data):
    return np.cov(data, rowvar=0)


def get_eigenvalues(data):
    return np.linalg.eig(get_covariance_matrix(data))[0]


def get_eigenvectors(data):
    return np.linalg.eig(get_covariance_matrix(data))[1]


def get_eigenvectors_after_kaiser_rule(data):
    #  The Kaiser rule is to drop all components with eigenvalues under 1.0
    eigenvectors = get_eigenvectors(data)
    eigenvalues = get_eigenvalues(data)
    keepers = []
    for i in range(len(eigenvalues)):
        if eigenvalues[i] > 1:
            keepers.append(i)
    return np.take(eigenvectors, keepers, axis=0)


def get_mapped_points(data, output_dim=1):
    eigenvectors_t = np.transpose(get_eigenvectors(data))
    return np.array(
        [
            values[:output_dim]
            for values in np.array(list(map(lambda x: np.dot(eigenvectors_t, x), data)))
        ]
    )


training_data = np.array([[0, 0], [4, 0], [2, 1], [6, 3]])
print("Mean vector: \n{}\n".format(get_mean_vector(training_data)))
print("Covariance matrix: \n{}\n".format(get_covariance_matrix(training_data)))
print("Eigenvalues: \n{}\n".format(get_eigenvalues(training_data)))
print("Eigenvectors: \n{}\n".format(get_eigenvectors(training_data)))
print(
    "Eigenvectors kaisers rule: \n{}\n".format(
        get_eigenvectors_after_kaiser_rule(training_data)
    )
)
print("Mapped points: \n{}\n".format(get_mapped_points(training_data)))
