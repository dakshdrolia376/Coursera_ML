
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt
import scipy.io as scio
import math
from sklearn.svm import SVC
from PIL import Image

# just defined a function to use hold on in matlab
def holdOn():
	input('\n\nPress Enter  to continue....')
# defining sigmoid function for a float x
def sigmoid(x):
	return 1/(1+np.exp(-1*x))

def findClosestCentroids(X, centroids):
       K = centroids.shape[0]
       idx = np.zeros((X.shape[0],1))
       for i in range(idx.size):
              idx[i] = np.argmin(np.sum((centroids-X[i])**2,axis = 1)) + 1
       return idx
def computeCentroids(X, idx, K):
       m, n = X.shape
       centroids = np.zeros((K,n))
       for i in range(K):
              pos = np.where( idx == i+1 )[0]
              centroids[i] = np.mean(X[pos, :],axis = 0) 
       return centroids
def plotDataPoints(X, idx, K):
       for i in range(K):
              pos = np.where( idx == i+1 )[0]
              if i == 0:
                     plt.scatter(X[pos,0], X[pos,1], color = 'r', marker = 'o')
              if i == 1:
                     plt.scatter(X[pos,0], X[pos,1], color = 'g', marker = 'o')
              if i == 2:
                     plt.scatter(X[pos,0], X[pos,1], color = 'b', marker = 'o')
              if i == 3:
                     plt.scatter(X[pos,0], X[pos,1], color = 'm', marker = 'o')
def plotProgresskMeans(X, centroids, previous, idx, K, i):
       plotDataPoints(X, idx, K)
       plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', color = 'black')
       
       
def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
       m, n = X.shape
       K = initial_centroids.shape[0]
       centroids = initial_centroids
       previous_centroids = centroids
       idx = np.zeros((m, 1))
       for i in range(max_iters ):
              print ('K - means Iteration {}/{}......\n'.format(i+1, max_iters))
              idx = findClosestCentroids(X, centroids)
              centroids = computeCentroids(X, idx, K)
       if plot_progress:
              plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
       return centroids, idx
def kMeansInitCentroids(X, K):
       ind = np.random.randint(0,X.shape[0],size = K)
       print(ind)
       return X[ind,:]
# Find closest Centroids
print ('Finding closest centroids  \n\n')
data = scio.loadmat('ex7/ex7data2.mat')
X = data['X']
K = 3         # no of centroids 
initial_centroids = np.array([[3, 3],[6, 2],[8, 5]])
idx = findClosestCentroids(X, initial_centroids)

# Compute means
print ('Computing Centroid means\n')
centroids = computeCentroids(X, idx, K)

# K-means Clustering
print ('Running K-Means clustering on example dataset. \n \n')
data2 = scio.loadmat('ex7/ex7data2.mat')
K = 3
max_iters = 10
initial_centroids = np.array([[3, 3],[6, 2],[8, 5]])
centroids, idx = runkMeans(X, initial_centroids, max_iters, False)
print ('\nK-means Done...\n\n')

# K-means clustering on pixels
print ('Running K-means clustering on pixels from an image.\n\n')
img = Image.open('ex7/bird_small.png')
A = np.array(img, dtype = float)
A = A/255
img_size = A.shape
X = np.reshape(A, (img_size[0]*img_size[1], 3))
K = 16
max_iter = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iter)

# Image Compression
print ('Applying K-means to compress an image.\n\n')
idx = findClosestCentroids(X, centroids)
idx = np.array(idx, dtype = int)
X_recovered = centroids[idx-1, :]
X_recovered = np.reshape(X_recovered ,( img_size[0], img_size[1], 3))
fig1 = plt.figure()
ax1 = fig1.add_subplot('121')
ax1.imshow(A)
ax2 = fig1.add_subplot('122')
ax2.imshow(X_recovered)
fig1.show()