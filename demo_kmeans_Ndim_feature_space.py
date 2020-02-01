## import libraries
from copy import deepcopy

import numpy as np
from numpy import linalg

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

## initialisation algorithm
def func_Kmeans(X, ncls, init='k-means++', max_iter=300):

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

        # plot(X, centroids[:0+1, :], 0, ncls)

        ## compute remaining k - 1 centroids
        for icls in range(1, ncls):

            distances = np.zeros((icls, ndata))
            dist_min = np.zeros(ndata)

            ## initialize a list to store distances of data
            ## points from nearest centroid
            for icls_pre in range(icls):
                distances[icls_pre, :] = linalg.norm(X - centroids[icls_pre, :], axis=1)

            idx_min = np.asarray(range(0, icls, icls * ndata)) + np.argmin(distances, axis=0)

            for imin in range(len(idx_min)):
                dist_min[imin] = distances[idx_min[imin], imin]

            imax = np.argmax(dist_min) % ndata
            centroids[icls, :] = X[imax, :]

            # plot(X, centroids[:icls+1, :], icls, ncls)

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

## Generate dataset
# Creating a iris dataset with 3 clusters and 4 features
# data = load_iris()
# X = data['data']    # The number of data with C features (C = 4)
# Y = data['target']  # The number of classes (Y = 3)

# Creating a sample dataset with 5 clusters and 3 features
X, Y = make_blobs(n_samples=500, n_features=5, centers=3)
ncls = 3                # The number of classes (clusters)
ndata = X.shape[0]      # The number of training data
nfeature = X.shape[1]   # The number of features in the data

init = 'k-means++'      # 'k-means++', 'random'
max_iter = 300

## Elbow method to choose the optimap number of clusters
fig = plt.figure(1)

cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.25, 0.0, 0.0],
                   [0.75, 1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.5,  0.0, 0.0],
                   [1.0,  1.0, 1.0]]}

newcmp = LinearSegmentedColormap('ClusterCmap', segmentdata=cdict, N=11)
rgba = newcmp(np.linspace(0, 1, 11))

wcss = []

for icls in range(1, 12):

    Kmean = KMeans(n_clusters=icls, init=init, max_iter=max_iter)
    Kmean.fit(X)

    centers = Kmean.cluster_centers_
    clusters = Kmean.labels_

    Y_icls = Y
    C = centers

    if (icls + 1) % 2 == 0:
        # ax = fig.add_subplot(2, 3, (icls + 1) // 2, projection='3d')
        #
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_icls)
        # ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=rgba[:icls, :], s=300)

        ax = fig.add_subplot(2, 3, (icls + 1) // 2)

        ax.scatter(X[:, 0], X[:, 1], c=Y_icls)
        ax.scatter(C[:, 0], C[:, 1], marker='*', c=rgba[:icls, :], s=300)

        plt.grid(b=True, which='both', color='k', linestyle='--', linewidth=0.5)

        ax.set_title('K-means++ (%d clusters)' % icls)
        ax.set_xlabel('1st feature')
        ax.set_ylabel('2nd feature')

    wcss.append(Kmean.inertia_)
plt.tight_layout()

fig = plt.figure(2)
plt.plot(range(1, 12), wcss, color='r', linestyle='-', linewidth=2, marker='o')
plt.grid(b=True, which='both', color='k', linestyle='--', linewidth=0.5)
plt.title('Elbow Method; The optimal number of clusters = 3')
plt.xlabel('The number of clusters')
plt.ylabel('Within Cluster Sum of Squares (WCSS)')

plt.xticks([i for i in range(1, 12)], ["{} clusters".format(i + 1) for i in range(11)])
plt.xlim((0, 12))
plt.tight_layout()

## Plot the ground truch
fig = plt.figure(3)

ax = fig.add_subplot(131, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
plt.tight_layout()

ax.set_title('Ground Truth')
ax.set_xlabel('1st feature')
ax.set_ylabel('2nd feature')
ax.set_zlabel('3rd feature')

## K-means in sci-kit Learn
Kmean = KMeans(n_clusters=ncls, init=init, max_iter=max_iter)
Kmean.fit(X)
centers = Kmean.cluster_centers_
clusters = Kmean.labels_

# Label from dataset
Y_clc = Y
C = centers

# Predict from Kmeans in sci-kit Learn
Y_pred_sci = clusters
C = centers

# Plot the data along the classes
fig = plt.figure(3)

ax = fig.add_subplot(132, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_pred_sci)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=rgba[:ncls, :], s=1000)
plt.tight_layout()

ax.set_title('K-means in Sci-kit (%d clusters)' % ncls)
ax.set_xlabel('1st feature')
ax.set_ylabel('2nd feature')
ax.set_zlabel('3rd feature')


## K-means from scratch
# Generate random centers, here we sigma and mean to ensure it represent the whole data
clusters, centers = func_Kmeans(X, ncls, init=init, max_iter=max_iter)

# Predict from Kmeans from scratch
Y_pred_scratch = clusters
C = centers

# Plot the data along the classes
fig = plt.figure(3)

ax = fig.add_subplot(133, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_pred_scratch)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=rgba[:ncls, :], s=1000)
plt.tight_layout()

ax.set_title('K-means from Scratch (%d clusters)' % ncls)
ax.set_xlabel('1st feature')
ax.set_ylabel('2nd feature')
ax.set_zlabel('3rd feature')

plt.show()
