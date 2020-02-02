## import libraries
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

## Set colormap
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

## Generate dataset
# Creating a iris dataset with 3 clusters and 4 features
# data = load_iris()
# X = data['data']    # The number of data with C features (C = 4)
# Y = data['target']  # The number of classes (Y = 3)

# Creating a sample dataset with 5 clusters and 3 features
X, Y = make_blobs(n_samples=500, n_features=5, centers=3)

## User defined parameters
ndata = X.shape[0]      # The number of training data
nfeature = X.shape[1]   # The number of features in the data

ncls = 3                # The number of classes (clusters)

init = 'k-means++'      # 'k-means++', 'random'
max_iter = 300

## Elbow method to choose the optimap number of clusters
fig = plt.figure(1)

wcss = []

for icls in range(1, 12):

    Kmean = KMeans(n_clusters=icls, init=init, max_iter=max_iter)
    Kmean.fit(X)

    centers = Kmean.cluster_centers_
    clusters = Kmean.labels_

    Y_icls = clusters
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
plt.show()