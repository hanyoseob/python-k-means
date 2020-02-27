import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from numpy import linalg

## initialisation algorithm
def func_Kmeans(X, ncls, init='k-means++', max_iter=300, bfig=False):

    def plot(X, centroids, icls, ncls):
        plt.figure(10)
        plt.subplot(1, ncls, icls + 1)

        plt.scatter(X[:, 0], X[:, 1], marker='.',
                    color='gray', label='data points')
        plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                    color='black', label='previously selected centroids')
        plt.scatter(centroids[-1, 0], centroids[-1, 1],
                    color='red', label='next centroid')
        plt.title('Select % d th centroid' % (centroids.shape[0]))

        plt.legend()
        # plt.xlim(-5, 12)
        # plt.ylim(-10, 15)
        # plt.show()

    def func_Kmeanspp(X, ncls):
        '''
        intialized the centroids for K-means++
        '''

        # The number of training data
        ndata = X.shape[0]

        # The number of features in the data
        nfeature = X.shape[1]

        ## initialize the centroids list and add
        ## a randomly selected data point to the list
        centroids = np.zeros((ncls, nfeature))
        centroids[0, :] = X[np.random.randint(ndata), :]

        if bfig:
            plot(X, centroids[:0+1, :], 0, ncls)

        ## compute remaining k - 1 centroids
        for icls in range(1, ncls):

            distances = np.zeros((icls, ndata))
            dist_min = np.zeros(ndata)

            ## initialize a list to store distances of data
            ## points from nearest centroid
            for icls_pre in range(icls):
                distances[icls_pre, :] = linalg.norm(X - centroids[icls_pre, :], axis=1)

            idx_min = np.argmin(distances, axis=0)

            for imin in range(len(idx_min)):
                dist_min[imin] = distances[idx_min[imin], imin]

            imax = np.argmax(dist_min) % ndata
            centroids[icls, :] = X[imax, :]

            if bfig:
                plot(X, centroids[:icls+1, :], icls, ncls)

        return centroids

    # The number of training data
    ndata = X.shape[0]

    # The number of features in the data
    nfeature = X.shape[1]

    if init == 'ramdom':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        centers_init = np.random.randn(ncls, nfeature)*std + mean
    else:
        centers_init = func_Kmeanspp(X, ncls)

    # Store new centers
    centers = deepcopy(centers_init)

    clusters = np.zeros(ndata)
    distances = np.zeros((ndata, ncls))

    # When, after an update, the estimate of that center stays the same, exit loop
    for i in range(max_iter):
    # while error != 0:
        # Measure the distance to every center
        for icls in range(ncls):
            distances[:, icls] = linalg.norm(X - centers[icls], axis=1)

        # Assign all training data to closest center
        clusters = np.argmin(distances, axis=1)

        centers_pre = deepcopy(centers)

        # Calculate mean for every cluster and update the center
        for icls in range(ncls):
            if (clusters == icls).any():
                centers[icls, :] = np.mean(X[clusters == icls], axis=0)

        error = linalg.norm(centers - centers_pre)

        if error == 0:
            break

    return clusters, centers
