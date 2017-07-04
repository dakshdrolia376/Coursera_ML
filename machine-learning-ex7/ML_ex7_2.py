from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt
import scipy.io as scio
import math
from sklearn.svm import SVC
from PIL import Image
import numpy as np

# just defined a function to use hold on in matlab
def holdOn():
	input('\n\nPress Enter  to continue....')
# defining sigmoid function for a float x
def sigmoid(x):
	return 1/(1+np.exp(-1*x))

def featureNormalize(X):
       mu = np.mean(X, axis = 0)
       mu = mu.reshape(1,mu.size)
       X_norm = X - mu
       sigma = np.std(X_norm, axis = 0).reshape(1,X.shape[1])
       X_norm = X_norm / sigma
       return X_norm, mu, sigma
def pca(X):
       m, n = X.shape
       sigma = (1.0/m)*(X.T.dot(X))
       u,s,v = np.linalg.svd(sigma)
       return u,s
def projectData(X, U ,K):
       U = U[:, range(K)]
       Z = X.dot(U)
       return Z

# Load Example Dataset
print ('Visualizing example datset for PCA.\n\n')
data = scio.loadmat('ex7/ex7data1.mat')
X = data['X']
fig1 = plt.figure()
ax1 = fig1.add_subplot('111')
ax1.scatter(X[:, 0], X[:, 1], marker = 'o')
fig1.show()

# Principal Component Analysis
print ('\nRunning PCA on example dataset\n\n')
X_norm, mu, sigma = featureNormalize(X)
U,S = pca(X_norm)

# Dimension reduction
print ('Dimension reduction on example dataset.')
fig2 = plt.figure()
ax2 = fig2.add_subplot('111')
ax2.scatter(X_norm[:, 0], X_norm[:, 1], marker = 'o')
fig2.show()
K = 1
Z = projectData(X_norm, U, K)
X_rec = U[:, range(K)].dot(Z.T).T
fig3 = plt.figure()
ax3 = fig3.add_subplot('111')
ax3.scatter(X_rec[:, 0], X_rec[:, 1], marker = 'o')
ax3.scatter(X_norm[:, 0], X_norm[:, 1], marker = 'x', color = 'red') 
for i in range(X.shape[0]):
       ax3.plot(np.array([X_rec[i,0],X_norm[i,0]]),np.array([X_rec[i,1],X_norm[i,1]]),ls = '--',color = 'red')
fig3.show()
# Loading and Visualizing Face Data
print ('\nLoading face Dataset.\n')
face_data = scio.loadmat('ex7/ex7faces.mat')
X = face_data['X']
X_temp = X[range(10),:]
a = X_temp.size
a =  (int(a/32))
X_temp = X_temp.reshape(a, 32)
fig4 = plt.figure()
ax4 = fig4.add_subplot('111')
ax4.imshow(X_temp.T,cmap = 'gray')

# PCA on Face Data
print ('Running PCA on face Data\n')
X_norm, mu, sigma = featureNormalize(X)
U,S = pca(X_norm)
U_temp = U[:,range(36)].T
fig5 = plt.figure()
a = U_temp.size
a =  (int(a/32))
U_temp = U_temp.reshape(a, 32)
ax5 = fig5.add_subplot('111')
ax5.imshow(U_temp.T[:,range(32*10)],cmap = 'gray')

# Dimension Reduction for Faces
print ('\n Dimension reduction for face dataset.\n')
K = 100
Z = projectData(X_norm, U, K)
print(Z.shape)
# Visualization Of faces after after PCA dimension reduction
print ('Visualizing the projected( reduced dimension) faces.\n\n')
K = 100
X_rec  = U[:, range(K)].dot(Z.T).T
print(X_rec.shape)
X_temp = X_norm[range(10),:]
a = X_temp.size
a =  (int(a/32))
X_temp = X_temp.reshape(a, 32)
fig6 = plt.figure()
ax6 = fig6.add_subplot('211')
ax6.imshow(X_temp.T,cmap = 'gray')
X_temp = X_rec[range(10),:]
a = X_temp.size
a =  (int(a/32))
X_temp = X_temp.reshape(a, 32)
fig6 = plt.figure()
ax6 = fig6.add_subplot('212')
ax6.imshow(X_temp.T,cmap = 'gray')
fig6.show()