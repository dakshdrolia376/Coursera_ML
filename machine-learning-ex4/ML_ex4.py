import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt
import scipy.io as scio
import pandas as pd

def holdOn():
    raw_input ('Press Enter to continue.....\n')

def sigmoid(x):
    return (1.0/(1+np.exp(-1*x)))

def sigmoidGrad(x):
    return (sigmoid(x))*(1 - sigmoid(x))

def nnGradFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmbda):
    m = y.size
    Delta0 = np.zeros((hidden_layer_size,input_layer_size+1))
    Delta1 = np.zeros((num_labels,hidden_layer_size+1))
    Theta0 = np.array(nn_params[0:((input_layer_size + 1)*(hidden_layer_size))].reshape(hidden_layer_size,input_layer_size+1))
    Theta1 = np.array(nn_params[((input_layer_size + 1)*(hidden_layer_size)):].reshape(num_labels,hidden_layer_size+1))
    h_theta = np.zeros((m,num_labels))
    y_matrix = pd.get_dummies(y.ravel()).as_matrix()
    for i in range(m):
        a0 = X[i].reshape(X.shape[1],1)
        z1 = Theta0.dot(np.concatenate((np.array([[1]]),a0),axis = 0))
        a1 = sigmoid(z1)
        z2 = Theta1.dot(np.concatenate((np.array([[1]]),a1),axis = 0))
        a2 = sigmoid(z2)
        h_theta[i] = a2.ravel()
        delta2 = a2 - y_matrix[i].reshape(num_labels,1)
        delta1 = Theta1[:,1:].T.dot(delta2)*(a1*(1-a1))
        Delta1 = Delta1 + delta2.dot(np.concatenate((np.array([[1]]),a1),axis = 0).T)
        Delta0 = Delta0 + delta1.dot(np.concatenate((np.array([[1]]),a0),axis = 0).T)
    D1 = (1.0/m)*(Delta1) + lmbda*(Theta1)
    D0 = (1.0/m)*(Delta0) + lmbda*(Theta0)
    D1 = np.concatenate(((1.0/m)*(Delta1[:,0].reshape(Delta1.shape[0],1)),D1[:,1:]),axis = 1)
    D0 = np.concatenate(((1.0/m)*(Delta0[:,0].reshape(Delta0.shape[0],1)),D0[:,1:]),axis = 1)
    return np.concatenate((D0.ravel(),D1.ravel()))

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmbda):
    m = y.size
    Theta0 = np.array(nn_params[0:((input_layer_size + 1)*(hidden_layer_size))].reshape(hidden_layer_size,input_layer_size+1))
    Theta1 = np.array(nn_params[((input_layer_size + 1)*(hidden_layer_size)):].reshape(num_labels,hidden_layer_size+1))
    h_theta = np.zeros((m,num_labels))
    y_matrix = pd.get_dummies(y.ravel()).as_matrix()
    for i in range(m):
        a0 = X[i].reshape(X.shape[1],1)
        z1 = Theta0.dot(np.concatenate((np.array([[1]]),a0),axis = 0))
        a1 = sigmoid(z1)
        z2 = Theta1.dot(np.concatenate((np.array([[1]]),a1),axis = 0))
        a2 = sigmoid(z2)
        h_theta[i] = a2.ravel()
    J = ((-1.0/m)*(np.sum(np.log(h_theta)*y_matrix + np.log(1-h_theta)*(1.-y_matrix))))+((lmbda/(2.0*m))*(np.sum(Theta1[:,1:]**2) + np.sum(Theta0[:,1:]**2)))
    return J  
    
        
        

def  debugInitializeWeights(fan_out , fan_in):
    W = np.zeros((fan_out , 1 + fan_in))
    W = np.reshape(np.sin(np.arange(1,W.size + 1)),(fan_out,(1 + fan_in)))
    return W
def checkingBp(lmbda):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights(hidden_layer_size , input_layer_size)
    Theta2 = debugInitializeWeights(num_labels , hidden_layer_size)
    
    X = debugInitializeWeights(m,input_layer_size - 1)
    y = 1 + ( np.arange(1,m+1) % num_labels )
    y = y.reshape(m,1)
    nn_params = np.concatenate((Theta1.ravel(),Theta2.ravel()))
    cost = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmbda)
    grad = nnGradFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmbda)
    #final_nn_params = scopt.fmin_bfgs(nnCostFunction,x0 = nn_params ,fprime = nnGradFunction,args = (input_layer_size,hidden_layer_size,num_labels,X,y,lmbda),maxiter = 10)
    print cost , grad

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# Loading data
print 'Loading Data'
data = scio.loadmat('ex4/ex4data1.mat')
X = data['X']
y = data['y']
m = X.shape[0]

# Loading Parameters
print '\nLoading Saved Neural Network Parameters.....\n'
weights = scio.loadmat('ex4/ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
nn_params = np.concatenate((Theta1.ravel(),Theta2.ravel()))

# Compute Cost (Feed Forward)---without regularization
print '\n Feedforward using Neural Network.....\n'
lmbda = 0
J= nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmbda)
print 'Cost without regularization: %f'%(J)

# Compute Cost(Feed Forward) ---with regularization
lmbda = 1.0
J= nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmbda)
print 'Cost with regularization and parameter lambda 1: %f'%(J)

# Sigmoid Gradient
g = sigmoidGrad(np.array( [-1 ,-0.5, 0, 0.5, 1]))
print 'Sigmoid Gradient is :' 
print g

# Initializing Parameters
initial_theta1 = np.random.rand(hidden_layer_size,input_layer_size+1)
initial_theta2 = np.random.rand(num_labels,hidden_layer_size+1)
initial_nn_params = np.concatenate((initial_theta1.ravel(),initial_theta2.ravel()))

# Implement Backpropagation
print'\nChecking Backpropagation......\n'
lmbda = 0
checkingBp(lmbda)