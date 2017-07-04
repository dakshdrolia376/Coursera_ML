import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt
import scipy.io as scio
import math
from sklearn.svm import SVC

# just defined a function to use hold on in matlab
def holdOn():
	input('\n\nPress Enter  to continue....')
# defining sigmoid function for a float x
def sigmoid(x):
	return 1/(1+np.exp(-1*x))
def plotData(X , y):
       pos = np.where(y.ravel() == 1)
       neg = np.where(y.ravel() == 0)
       plt.scatter(X[pos,0],X[pos,1],marker = '+' , c = 'g')
       plt.scatter(X[neg,0],X[neg,1],marker = 'o' , c = 'r')

def plot_svc(svc , X, y, h = 0.02, pad = 0.25):
       x_min , x_max = X[:, 0].min() - pad , X[: , 0].max() + pad
       y_min , y_max = X[:, 1].min() - pad , X[: , 1].max() + pad
       xx , yy = np.meshgrid(np.arange(x_min ,x_max ,h), np.arange(y_min, y_max,h))
       Z = svc.predict(np.c_[xx.ravel(),yy.ravel()])
       Z = Z.reshape(xx.shape)
       plt.contourf(xx,yy,Z,cmap = plt.cm.Paired , alpha = 0.2)
       plotData(X, y)
       sv = svc.support_vectors_
       plt.scatter(sv[:,0],sv[:,1],c = 'k' , marker = '|' )
       plt.xlim(x_min , x_max)
       plt.ylim(y_min , y_max)
       plt.show()
def gaussianKernel(x1, x2, sigma):
       x1 = x1.reshape(x1.size,1)
       x2 = x2.reshape(x1.size,1)
       sim = 0
       sim = np.sum((x1 - x2)**2)/(2.0*(sigma**2))
       sim = math.exp(-1*sim)
       return sim
def dataset3Params(X3, y3, Xval, yval):
       mean = 0;
       for c in np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
              for sigma in np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
                     g = (1.0/(2*(sigma**2)))
                     clf = SVC(C = c,gamma = g, kernel = 'rbf')
                     clf.fit(X3,y3)
                     Z = clf.predict(Xval).reshape(yval.shape)
                     
                     if mean <  np.mean(np.array(Z == yval , dtype = float)):
                            final_C = c
                            final_sigma = sigma
                            mean = np.mean(np.array( Z == yval , dtype = float))
                            print (mean)
       return final_C,final_sigma
       
#==============================================================================
# # Loading and visualizing Data
# print ('Loading and visualizing data')
# data = scio.loadmat('ex6/ex6data1.mat')
# X = data['X']
# y = data['y']
# plotData(X, y)
#==============================================================================

#==============================================================================
# # Training linear SVM
# data1 = scio.loadmat('ex6/ex6data1.mat')
# print ('\nTraining Linear SVM.......\n')
# clf = SVC(C = 1 , kernel = 'linear')
# clf.fit(X, y.ravel())
# plot_svc(clf , X, y)
#==============================================================================

#==============================================================================
# # Gaussian Kernel
# print (' Evaluating the Gaussian Kernel.....!\n' )
# x1 = np.array([1 ,2 ,1])
# x2 = np.array([0 ,4 ,-1])
# sigma = 2
# sim = gaussianKernel(x1, x2, sigma)
# print (sim)
#==============================================================================

#==============================================================================
# # Visualizing dataset2
# data2 = scio.loadmat('ex6/ex6data2.mat')
# X2 = data2['X']
# y2 = data2['y']
# plotData(X2,y2)
#==============================================================================

#==============================================================================
# # Training SVM with RBF Kernel
# print ('\n Training SVM with RBF Kernel (this may take 1 to 2 minutes)....\n')
# data2 = scio.loadmat('ex6/ex6data2.mat')
# X2 = data2['X']
# y2 = data2['y']
# clf = SVC(C = 1, kernel = 'rbf' , gamma = 50.0)
# clf.fit(X2,y2)
# plot_svc(clf , X2 , y2)
# 
#==============================================================================

#==============================================================================
# # Visualizing dataset 03
# print ('Loading and Visualizing data.....\n')
# data3 = scio.loadmat('ex6/ex6data3.mat')
# X3 = data3['X']
# y3 = data3['y']
# plotData(X3, y3)
#==============================================================================

#==============================================================================
# # Training SVM with RBF Kernel (Dataset 03)
# data3 = scio.loadmat('ex6/ex6data3.mat')
# X3 = data3['X']
# y3 = data3['y']
# Xval = data3['Xval']
# yval = data3['yval']
# c, sigma = dataset3Params(X3, y3, Xval, yval)
# g = (1.0/(2*(sigma**2)))
# clf = SVC(C = c , gamma = g, kernel = 'rbf', degree = 3)
# clf.fit(X3,y3)
# plot_svc(clf, X3, y3)
#==============================================================================
