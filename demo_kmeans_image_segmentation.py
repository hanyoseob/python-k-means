## import libraries
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

from func_Kmeans import func_Kmeans
from skimage import transform

## Generate dataset
IMG = plt.imread('test_image.jpg')
IMG = np.uint8(255*transform.resize(IMG, [512, 768]))
X = IMG.reshape((-1, 3))

## User defined parameters
ndata = X.shape[0]      # The number of training data
nfeature = X.shape[1]   # The number of features in the data

ncls = 2                # The number of classes (clusters)
# ncls = 3                # The number of classes (clusters)
# ncls = 4                # The number of classes (clusters)

init = 'k-means++'      # 'k-means++', 'random'
max_iter = 300

## Plot the ground truth
fig = plt.figure(1)

ax = fig.add_subplot(131)
plt.imshow(IMG)
ax.set_title('Ground Truth')
plt.axis('off')

## K-means in sci-kit Learn
Kmean = KMeans(n_clusters=ncls, init=init, max_iter=max_iter)
Kmean.fit(X)
centers = Kmean.cluster_centers_
clusters = Kmean.labels_

Y_pred_sci = np.uint8(centers[clusters]).reshape(IMG.shape)

# Plot the data along the classes
fig = plt.figure(1)

ax = fig.add_subplot(132)
plt.imshow(Y_pred_sci)
ax.set_title('K-means in Sci-kit (%d clusters)' % ncls)
plt.axis('off')

## K-means from scratch
clusters, centers = func_Kmeans(X, ncls, init=init, max_iter=max_iter)

Y_pred_scratch = np.uint8(centers[clusters]).reshape(IMG.shape)

# Plot the data along the classes
fig = plt.figure(1)

ax = fig.add_subplot(133)
plt.imshow(Y_pred_scratch)
ax.set_title('K-means from Scratch (%d clusters)' % ncls)
plt.axis('off')

plt.show()


