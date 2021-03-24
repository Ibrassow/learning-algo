# -*- coding: utf-8 -*-
"""
@author: Ibrahima S. Sow

The algorithm has been written based on Andrew Ng's lecture notes available here :

http://cs229.stanford.edu/notes2020spring/cs229-notes7a.pdf

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random



class Kmeans:
    
    
    def __init__(self, X, k_clusters):
        self.X = X
        self.N = len(X)
        self.idx_cluster = np.zeros(self.N)
        self.k = k_clusters
        self.centroids = np.array([])
        self.cluster_data = {}
        self.tolerance = 1E-10
        self.__maxIter = 200
        
        

    def set_initial_centroids(self):
        
        """
        Select k random points from the dataset as initial centroids
        
        """
        idx = np.random.choice(self.N, self.k, replace=False)
        
        self.centroids = self.X[idx,:]
        
        for i in range(self.k):
            self.cluster_data[i] = np.empty((0,len(self.centroids[1])))
            
            
        print("initial centroids : {}\n".format(self.centroids))
        
        
         

    
    
    def update_centroids(self):
        """
        Update each centroids by averaging the points belonging to each cluster
        
        """
        print("old", self.centroids)
        for i in range(self.k):
            self.centroids[i] = np.mean(self.cluster_data[i], axis = 0)
        print("new", self.centroids)
        
        return self.centroids
    
    
    def dist_to_centroid(self, x1, x2):
        """
        Computes the Euclidian distance between two points

        """
        if len(x1) != len(x2):
            raise Exception("Can't compute the Euclidian distance : points do not have same dimensions")

        return distance.euclidean(x1, x2)
    
    
    def perform_clustering(self):
        """
        Returns
        -------
        idx_cluster : array containing cluster assignment of each data point

        """
        self.set_initial_centroids()
        
        
        error = 555
        iterations = 0
        
        while np.abs(error) > self.tolerance:
            
            for i in range(self.N):
                
                #Add the point to the nearest centroid
                #dist = [self.dist_to_centroid(self.X[i], self.centroids[idx]) for idx in self.centroids]
                dist = [self.dist_to_centroid(self.X[i], self.centroids[k]) for k in range(self.k)]
                cluster = dist.index(min(dist))
                #Update the indices array
                self.idx_cluster[i] = cluster
                self.cluster_data[cluster] = np.vstack((self.cluster_data[cluster], self.X[i]))
                
                
                
            previous_centroids = np.copy(self.centroids)
            self.update_centroids()
            
            error = np.sum((self.centroids - previous_centroids)/previous_centroids)
            
            iterations += 1
            print("Iteration {} - error : {}".format(iterations, error))
            
            if iterations >= self.__maxIter:
                print("Maximum number of iterations reached")
                break
            

        return self.idx_cluster
                

    
if __name__ == "__main__":
    

    
    """for testing purpose"""
    
    def generate_point(mean_x, mean_y, deviation_x, deviation_y):
        return [random.gauss(mean_x, deviation_x), random.gauss(mean_y, deviation_y)]
    
    
    cluster_mean_x = 100
    cluster_mean_y = 100
    cluster_deviation_x = 51
    cluster_deviation_y = 50
    point_deviation_x = 20
    point_deviation_y = 20

    n_clusters = 3
    points_per_cluster = 200
    
    
    cluster_centers = [generate_point(cluster_mean_x,
                                  cluster_mean_y,
                                  cluster_deviation_x,
                                  cluster_deviation_y)
                   for i in range(n_clusters)]
    

    X = np.array([generate_point(center_x,
                         center_y,
                         point_deviation_x,
                         point_deviation_y)
          for center_x, center_y in cluster_centers
          for i in range(points_per_cluster)])
     
    #plt.scatter(X[:,0], X[:,1])
    

    
    
    Classifier = Kmeans(X, n_clusters)
    idx = Classifier.perform_clustering()
    

    for i in range(len(idx)):
        plt.scatter(X[idx == 0, 0], X[idx == 0, 1], color="red", s=10, label="Cluster1")
        plt.scatter(X[idx == 1, 0], X[idx == 1, 1], color="blue", s=10, label="Cluster2")
        plt.scatter(X[idx == 2, 0], X[idx == 2, 1], color="green", s=10, label="Cluster3")
        

    
    
    
        
    
    
    
    
    

    
