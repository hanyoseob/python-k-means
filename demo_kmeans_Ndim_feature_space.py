## import libraries
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

from func_Kmeans import func_Kmeans

##
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

## Plot the ground truch
fig = plt.figure(1)

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
fig = plt.figure(1)

ax = fig.add_subplot(132, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_pred_sci)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=rgba[:ncls, :], s=1000)
plt.tight_layout()

ax.set_title('K-means in Sci-kit (%d clusters)' % ncls)
ax.set_xlabel('1st feature')
ax.set_ylabel('2nd feature')
ax.set_zlabel('3rd feature')


## K-means from scratch
clusters, centers = func_Kmeans(X, ncls, init=init, max_iter=max_iter)

# Predict from Kmeans from scratch
Y_pred_scratch = clusters
C = centers

# Plot the data along the classes
fig = plt.figure(1)

ax = fig.add_subplot(133, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_pred_scratch)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=rgba[:ncls, :], s=1000)
plt.tight_layout()

ax.set_title('K-means from Scratch (%d clusters)' % ncls)
ax.set_xlabel('1st feature')
ax.set_ylabel('2nd feature')
ax.set_zlabel('3rd feature')

plt.show()
