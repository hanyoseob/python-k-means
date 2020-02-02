## import libraries
import numpy as np

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from func_Kmeans import func_Kmeans

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
X, Y = make_blobs(n_samples=500, n_features=2, centers=3)

## User defined parameters
ndata = X.shape[0]      # The number of training data
nfeature = X.shape[1]   # The number of features in the data

ncls = 3                # The number of classes (clusters)

init = 'k-means++'      # 'k-means++', 'random'
max_iter = 300

## K-means from scratch
clusters, centers = func_Kmeans(X, ncls, init=init, max_iter=max_iter, bfig=True)

# Predict from Kmeans from scratch
Y_pred_scratch = clusters
C = centers

# Plot the data along the classes
fig = plt.figure(1)

ax = fig.add_subplot(1, 1, 1)

ax.scatter(X[:, 0], X[:, 1], c=Y_pred_scratch)
ax.scatter(C[:, 0], C[:, 1], marker='*', c=rgba[:ncls, :], s=1000)
plt.tight_layout()

ax.set_title('K-means from Scratch (%d clusters)' % ncls)
ax.set_xlabel('1st feature')
ax.set_ylabel('2nd feature')

plt.show()
