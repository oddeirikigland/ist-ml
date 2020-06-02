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
            print("For input x{} with value {}:  ".format(i+1,input[i]))
            dist_clusters = []
            for j in range(len(clusters)):
                dist_clusters.append(distance.euclidean(input[i], clusters[j]))
                print("\tDistance to cluster {} ({}):   {}".format(j+1, clusters[j], dist_clusters[j]))
            distance_dict[i] = dist_clusters
            assigned_cluster = dist_clusters.index(min(dist_clusters))
            cluster_dict[assigned_cluster].append(i)
        print("Clusters assigned following x-values:")
        print(cluster_dict)
        clusters = compute_new_clustercentroids(cluster_dict,input)




def compute_new_clustercentroids(cluster_dict, input):
    centroids = []
    for i in range(len(cluster_dict)):
        elements = []
        for el in range(len(cluster_dict[i])):
            elements.append(input[cluster_dict[i][el]])
        centroids.append(np.mean(np.array(elements), axis=0))
    print("New centroids:")
    for i in range(len(centroids)):
        print("\tCentroid {}: {}".format(i,centroids[i]))
    return centroids



training_data = [[0,0], [1,0],[0,2],[2,2]]

clusters = [[2,0], [2,1]]

k_means(training_data,clusters)
