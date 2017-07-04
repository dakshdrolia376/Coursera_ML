import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sc
import math


#just defined a function to use hold on in matlab
def holdOn():
	raw_input('\n\nPress Enter  to continue....')


#defining sigmoid function for a float x
def sigmoid(x):
	return 1/(1+np.exp(-1*x))

#takes feature x1 & feature x2 -> column vectors of size m  and give as output of matrix containing all feature mapped till degree 6 ex: [x1 x2 x1^2 x1x2 x2^2...]
def mapFeature(x1,x2):
	degree = 6
	m = x1.size
	x1=x1.reshape((m,1))
	x2=x2.reshape((m,1))
	final_X=np.ones((m,1))
	
	for i in range(1,degree+1):
		for j in range(i+1):
			final_X = np.concatenate((final_X,np.array((x1**(i-j))*(x2**j))),axis = 1)
	return final_X

#return cost , also calculate gradient but I have to minimize cost so I am not returning gradient matrix
def CostFunctionReg( theta,X,y,lmbda ):
    h_theta = sigmoid(X.dot(theta))
    m = y.size
    theta_new = np.array(theta[1:])
    cost = ((-1.0/m)*(y.T.dot(np.log(h_theta))+(1-y).T.dot(np.log(1-h_theta)))) + (lmbda/(2*m))*np.sum(theta_new**2) 
    return cost

def gradientDescent(X,y,theta,alpha,iterations,lmbda):

	m = y.size
	b = np.transpose(X)
	Jhist = np.arange(iterations,dtype=float)
	for i in range(iterations):
		h_theta = X.dot(theta)
		for j in range(m):
			h_theta[j] = sigmoid(h_theta[j])
		
		theta = theta - ((alpha/m)*(X.T.dot(h_theta - y) + ((lmbda/(1.0*m))*theta)))
		Jhist[i] =CostFunctionReg(theta,X,y,lmbda)
	return theta,Jhist

#loading data from the file
data = np.loadtxt('ex2/ex2data2.txt',delimiter=',')

#extracting features X as [x1 x2...] where x1 , x2 are column vectors
X = np.array(data[:,range(data.shape[1]-1)])	

#extracting y as [1 0 1 0.....] where y is a one d array	
y = np.array(data[:,data.shape[1]-1])		

# m are no. of training examples		
m = y.size			

# making y a column vector from a row vector								
y = y.reshape((m,1))								

#plotting data 
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
pos = np.where(y==1)
neg = np.where(y==0)
ax1.scatter(X[pos,0],X[pos,1],marker = 'o',c = 'b')
ax1.scatter(X[neg,0],X[neg,1],marker = 'x',c = 'r')
ax1.legend(['Not Rejected', 'Rejected' ])
ax1.set_xlabel('Microchip test 1')
ax1.set_ylabel('Microchip test 2')
fig1.show()

holdOn()


#mapping more features from availabes ones
X = mapFeature(np.array(X[:,0]),np.array(X[:,1]))

# defining initial parameters to start
initial_theta = np.zeros((X.shape[1],1))

# regularization parameter
lmbda = 1

#computing cost for initial theta matched with expected results coming correct
cost   = CostFunctionReg(initial_theta, X,y,lmbda)
print cost


holdOn()
alpha = 0.01

# this function isn't working correctly . I searched all documents on internet regarding it but can't make it work . This function uses an
# advanced optimization algo BFGS to minimize any function f(x,(*args)) with initial parameters x0 to be minimized.
final_theta = sc.fmin_bfgs(f = CostFunctionReg,x0 = initial_theta,args = (X,y,lmbda))
#final_theta,Jhist = gradientDescent(X,y,initial_theta,alpha,20000,lmbda)


#Plotting contour
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
    	z[i, j] = (mapFeature(np.array(u[i]), np.array(v[j])).dot(final_theta))
z = z.T
ax1.contour(u, v, z)
ax1.legend(['y = 1', 'y = 0', 'Decision boundary'])
fig1.show()
#fig1.savefig('Decision_boundary.png')

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(range(20000),Jhist)
#fig2.show()