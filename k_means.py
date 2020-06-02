import numpy as np
from scipy.spatial import distance


def k_means(input, clusters, epochs=3):

    for epo in range(epochs):
        print("\n \nEPOCH {} ".format(epo))
        distance_dict = {}
        cluster_dict = {}
        for i in range(len(clusters)):
            cluster_dict[i] = []
        for i in range(len(input)):
            print("For input x{} with value {}:  ".format(i + 1, input[i]))
            dist_clusters = []
            for j in range(len(clusters)):
                dist_clusters.append(distance.euclidean(input[i], clusters[j]))
                print(
                    "\tDistance to cluster {} ({}):   {}".format(
                        j + 1, clusters[j], dist_clusters[j]
                    )
                )
            distance_dict[i] = dist_clusters
            assigned_cluster = dist_clusters.index(min(dist_clusters))
            cluster_dict[assigned_cluster].append(i)
        print("Clusters assigned following x-values:")
        print(cluster_dict)
        clusters = compute_new_clustercentroids(cluster_dict, input)
    return clusters, cluster_dict


def compute_new_clustercentroids(cluster_dict, input):
    centroids = []
    for i in range(len(cluster_dict)):
        elements = []
        for el in range(len(cluster_dict[i])):
            elements.append(input[cluster_dict[i][el]])
        centroids.append(np.mean(np.array(elements), axis=0))
    print("New centroids:")
    for i in range(len(centroids)):
        print("\tCentroid {}: {}".format(i, centroids[i]))
    return centroids


def intra_cluster_euclidean_dist(data_points, clusters, cluster_dict):
    total_dist = 0
    for key, value in cluster_dict.items():
        cluster = clusters[key]
        for index in value:
            point = data_points[index]
            total_dist += distance.euclidean(point, cluster)
    return total_dist


def mean_inter_cluster_centroid_euclidean_dist(clusters):
    total_dist = 0
    for cluster in clusters:
        for compare_cluster in clusters:
            total_dist += distance.euclidean(cluster, compare_cluster)
    return total_dist / len(clusters) ** 2


training_data = [[1, 0, 0], [8, 8, 4], [3, 3, 0], [0, 0, 1], [0, 1, 0], [3, 2, 1]]

clusters_2, cluster_dict_2 = k_means(training_data, training_data[:2])
clusters_3, cluster_dict_3 = k_means(training_data, training_data[:3])

# this functions uses euclidean distance, in the lab they dont not square the distance

print(intra_cluster_euclidean_dist(training_data, clusters_2, cluster_dict_2))
print(intra_cluster_euclidean_dist(training_data, clusters_3, cluster_dict_3))

print(mean_inter_cluster_centroid_euclidean_dist(clusters_2))
print(mean_inter_cluster_centroid_euclidean_dist(clusters_3))
