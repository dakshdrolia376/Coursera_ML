import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt
import scipy.io as scio
import math

# just defined a function to use hold on in matlab
def holdOn():
	input('\n\nPress Enter  to continue....')

# defining sigmoid function for a float x
def sigmoid(x):
	return 1/(1+np.exp(-1*x))
def linearRegCostFunction(theta,X,y,lmbda):
    m = y.size
    h_theta = X.dot(theta)
    J =( (1.0/(2*m))*(np.sum((h_theta - y)**2)))+((lmbda/(2.0 * m))*np.sum(theta[1:]**2))
    return J
def linearRegGradFunction(theta,X,y,lmbda):
    m = y.size
    theta = theta.reshape(X.shape[1],1)
    h_theta = X.dot(theta)
    grad = np.zeros((theta.shape))
    grad = ((1.0/m)*(X.T.dot(h_theta - y))) + ((1.0*lmbda)/m)*(theta)
    grad[0] -= ((1.0 * lmbda)/m)*(theta[0])
    return grad.flatten()
def trainLinearReg(X, y, lmbda):
    initial_theta = np.ones((X.shape[1],1))
    final_theta = scopt.minimize(linearRegCostFunction ,initial_theta, args = (X,y,lmbda),jac=linearRegGradFunction)
    return final_theta.x
def learningCurve(X, y, Xval, yval, lmbda):
    m = X.shape[0]
    error_train = np.zeros((m,1))
    error_val = np.zeros((m,1))
    for i in range(1,m+1):
        theta = trainLinearReg(X[:i,:].reshape(i,X.shape[1]),y[:i,:].reshape(i,1),lmbda)
        error_train[i-1] = linearRegCostFunction(theta , X[:i,:], y[:i,:], 0.0)
        error_val[i-1] = linearRegCostFunction(theta , Xval, yval, 0.0)
    return error_train, error_val
def polyFeatures(X, p) :
    m = X.shape[0]
    X_poly = X
    for i in range(2,p+1):
        X_poly = np.concatenate((X_poly , X**i),axis = 1)
    return X_poly
def featureNormalize(X):
    mu = np.mean(X)
    X = X - mu
    sigma = np.std(X)
    X = X/float(sigma)
    return X,mu,sigma
# Loading and Visualizing Data
print ('Loading and Visualizing data....\n')
data = scio.loadmat('ex5/ex5data1.mat')
print (data.keys())
X = data['X']
y = data['y']
Xtest = data['Xtest']
ytest = data['ytest']
Xval = data['Xval']
yval = data['yval']
m = X.shape[0]
fig1 = plt.figure()
ax1 = fig1.add_subplot('111')
ax1.scatter(X,y,marker = 'x',c = 'r')
ax1.set_xlabel('Change in water level (x)')
ax1.set_ylabel('Water flowing out of the dam (y)')
fig1.show()
holdOn()

# Regularized Linear Regression Cost
theta = np.array([[1],[1]])
J = linearRegCostFunction(theta,np.concatenate((np.ones((m,1)),X),axis = 1),y,1) 
print (J)

# Regularized Linear Regression Gradient
grad = linearRegGradFunction(theta,np.concatenate((np.ones((m,1)),X),axis = 1),y,1)
print (grad)

# Train Linear Regression
lmbda = 1.0
final_theta = trainLinearReg(np.concatenate((np.ones((m,1)),X),axis = 1),y,1.0)
ax1.plot(X.ravel(),np.concatenate((np.ones((m,1)),X),axis = 1).dot(final_theta).ravel())
fig1.show()

# Learning Curve for Linear regression
lmbda = 0
error_train , error_val = learningCurve(np.concatenate((np.ones((m,1)),X),axis = 1), y, np.concatenate((np.ones((Xval.shape[0],1)),Xval),axis = 1), yval , lmbda)
fig2 = plt.figure()
ax2 = fig2.add_subplot('111')
ax2.plot(range(1,m+1),error_train,color = 'green')
ax2.plot(range(1,m+1),error_val,color = 'blue')
fig2.show()

# Feature Mapping for Polynomial Regression
p = 8
X_poly = polyFeatures(X, p)
X_poly , mu , sigma = featureNormalize(X_poly)
X_poly = np.concatenate((np.ones((m,1)),X_poly),axis = 1)

X_poly_test = polyFeatures(Xtest , p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / float(sigma)
X_poly_test = np.concatenate((np.ones((X_poly_test.shape[0],1)),X_poly_test),axis = 1)

X_poly_val = polyFeatures(Xval , p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / float(sigma)
X_poly_val = np.concatenate((np.ones((X_poly_val.shape[0],1)),X_poly_val),axis = 1)

# Learning curve for polynomial regression
theta = trainLinearReg(X_poly , y , 1)
error_train1 , error_val1 = learningCurve(X_poly , y , X_poly_val, yval, lmbda)
fig3 = plt.figure()
ax3 = fig3.add_subplot('111')
ax3.plot(range(1,m+1),error_train1,color = 'green')
ax3.plot(range(1,m+1),error_val1,color = 'blue')
fig3.show()
print (X)
# Plotting this polynomial plot
ax1.scatter(X,X_poly.dot(theta).ravel(), color = 'green',marker = 'x')
fig1.show()

# Can't train polynomial regression by fmin_bfgs , on internet scilpy learn library is used . 