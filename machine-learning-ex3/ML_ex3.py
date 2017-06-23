import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt
import scipy.io as scio

def holdOn():
    raw_input ('Press Enter to continue.....\n')

def sigmoid(x):
    return (1.0/(1+np.exp(-1*x)))

def lrCostFunction( theta,X,y,lmbda ):
    h_theta = sigmoid(X.dot(theta))
    m = y.size
    theta_new = np.array(theta[1:])
    cost = ((-1.0/m)*(y.T.dot(np.log(h_theta))+(1-y).T.dot(np.log(1-h_theta)))) + (lmbda/(2*m))*np.sum(theta_new ** 2) 
    return cost

def lrGradFunction (theta,X,y,lmbda):
    m = y.size
    h_theta = sigmoid(X.dot(theta.reshape(X.shape[1],1)))
    grad = (1.0/m)*(X.T.dot(h_theta-y)) + (lmbda/m)*theta.reshape(X.shape[1],1)
    grad[0] -= (lmbda/m)*theta.reshape(X.shape[1],1)[0]
    return grad.flatten()

def oneVsAll(X,y,num_labels,lmbda):
    m = X.shape[0]
    n = X.shape[1]
    final_theta = np.zeros((num_labels,n+1))
    X = np.concatenate((np.ones((m,1)),X),axis = 1)
    for i in range(1,num_labels+1):
        initial_theta = np.zeros((n+1,1))
        final_theta[i-1] = (scopt.minimize(lrCostFunction,initial_theta,args = (X,np.array(y==i , dtype = float),lmbda),jac = lrGradFunction)).x
    return final_theta
    
def predictOneVsAll(theta,X):
    m = X.shape[0]
    X = np.concatenate((np.ones((m,1)),X),axis = 1)
    p = np.argmax(sigmoid(X.dot(theta.T)),axis = 1) + 1
    return p
    
#loading data
input_layer_size = 400
num_labels = 10
print 'Loading an visualizing data....\n'
data = scio.loadmat('ex3/ex3data1.mat')
X = data['X']
y = data['y']

#visualizing
rand_indices = np.random.choice(X.shape[0],20)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1 = plt.imshow(X[rand_indices,:].reshape(-1,20).T,cmap = 'gray_r')
fig1.show()

holdOn()

#Cost function and grad function
print 'Testing lrcostFunction() with regularization'
theta_t = np.array([[-2],[-1],[1],[2]])
X_t = np.concatenate((np.ones((5,1)),np.arange(1,16).reshape((5,3),order = 'F')/10.0),axis = 1)
y_t = np.array(np.array([[1],[0],[1],[0],[1]]) >= 0.5 ,dtype = float)
lambda_t = 3.0
J = lrCostFunction(theta_t,X_t,y_t,lambda_t)
grad = lrGradFunction(theta_t,X_t,y_t,lambda_t)
print 'Confirmed cost and grads '

holdOn()

#One-vs-All training
print '\n\nTraining One-vs-All Logistic Regression'
lmbda = 0.1
final_theta = oneVsAll(X,y,num_labels,lmbda)

holdOn()

#Predict for One-Vs-All
pred = predictOneVsAll(final_theta,X)

print np.mean(np.array(pred==y.ravel() , dtype = float))*100





