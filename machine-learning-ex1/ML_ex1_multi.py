import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def featureNormalize(X):
	X_norm = X
	mu = np.zeros((1,X.shape[1]))
	sigma = np.zeros((1,X.shape[1]))
	mu = np.mean(X,axis = 0 )
	sigma = np.std(X,axis = 0)
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			X_norm[i][j] = (X_norm[i][j]-mu[j])/sigma[j]
	return X_norm,mu,sigma
def computeCostMulti(X,y,theta):
	 m = y.size
	 hyp = X.dot(theta)
	 a = hyp - y
	 b = a**2
	 c = sum(b)
	 return (1.0/2*m)*(c)
def gradientDescentMulti(X,y,theta,alpha,iterations):
	m = y.size
	Jhist = np.zeros((iterations,1))
	for i in range(iterations):
		hyp = X.dot(theta)
		a = hyp - y
		b = np.transpose(X)
		c = b.dot(a)
		theta = theta - ((alpha/m)*(c))
		Jhist[i] = computeCostMulti(X,y,theta)
	return theta,Jhist
def normEq(X,y):
	a = np.transpose(X)
	b = a.dot(X)
	c = np.linalg.pinv(b)
	d = c.dot(np.transpose(X))
	e = d.dot(y)
	return e
data = np.loadtxt('ex1/ex1data2.txt',delimiter=',')
X = np.array(data[:,range(data.shape[1]-1)])
y = np.array(data[:,2])
m = y.size
y = y.reshape((m,1))

#normalization
print '\n\n\n Normalizing Features'
X,mu,sigma = featureNormalize(X)
X = np.concatenate((np.ones((m,1)),X),axis=1)
raw_input('\n\n\nPress Enter to continue')

#Gradient Descent
print '\nRunning Gradient Descent.....'
alpha = 0.001
iterations = 1500
#initializing theta
theta = [[85000.0],[130.0],[-9000.0]]
theta,Jhist = gradientDescentMulti(X,y,theta,alpha,iterations)

#plotting the cost function vs iterations
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(iterations),Jhist)
fig.show()

#Normal equations
X = np.array(data[:,range(data.shape[1]-1)])
y = np.array(data[:,data.shape[1]-1])
X = np.concatenate((np.ones((m,1)),X),axis = 1)
theta_norm = normEq(X,y)
print 'Normal Equaton:'
print theta_norm
print 'Gradient Descent:'
print theta