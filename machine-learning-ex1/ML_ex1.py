import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sc

def warmupExercise():
	A = np.eye(5);
	return A

def computeCost(theta,X,y):
	m = y.size
	return  (1.0/(2*m))*np.sum(((X.dot(theta))-y.reshape((m,1)))**2) 

def gradientDescent(X,y,theta,alpha,iterations):
    m = y.size
    b = np.transpose(X)
    c = y.reshape((m,1))
    Jhist = np.arange(1500,dtype=float)
    for i in range(1500):
		a = X.dot(theta)
		d = a - c
		theta = theta - ((alpha/m)*(b.dot(d)))
		Jhist[i] = computeCost(theta,X,y)
    return theta,Jhist

print 'Running warmupExercise'
print '5 x 5 identity matrix: \n'
X=warmupExercise()
print X

raw_input('Press Enter to continue')

print '\n\n\nPlotting data....'
data = np.loadtxt('ex1/ex1data1.txt',delimiter=',')
X = np.array(data[:,0])
y = np.array(data[:,1])
m = y.size
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X,y,color='red',linewidth='2.5',)
fig1.show()

raw_input('Press Enter to continue')

X_exp = np.concatenate((np.ones((m,1)),np.array(data[:,0]).reshape((m,1))),axis=1)
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01
print 'Investing the cost function'
J = computeCost([[-1],[2]],X_exp,y)
print 'Confirmed Compute Cost working correcty'

raw_input('\n\nPress enter to continue....')

print '\nRunning Gradient Descent.....\n'
theta,Jhist = gradientDescent( X_exp, y, theta, alpha, iterations );
print Jhist

print '\n Plotting linear fit\n'
ax1.plot(X,X_exp.dot(theta),color = 'blue', linestyle = '-')

temp1 = np.array([1,3.5])
predict1 = temp1.dot(theta)
print 'For popuation = 35,000, we predict a profit of %f\n' % (predict1*10000)

temp2 = np.array([1,7])
predict2 = temp2.dot(theta)
print 'For popuation = 70,000, we predict a profit of %f\n' % (predict2*10000)

raw_input('\n\nPress enter to continue....')

print 'Getting cost function Vs no. of iteratons'

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(range(iterations),Jhist)
fig2.show()

raw_input('\n\nPress enter to continue....')

print 'Visualising J(theta0,theta1,....)'
theta0 = np.linspace(-10,10,100)
theta1 = np.linspace(-1,4,100)

J_vals = np.zeros((theta0.size,theta1.size))

for i in range(theta0.size):
	for j in range(theta1.size):
		theta_new = np.array([[theta0[i]],[theta1[j]]])
		J_vals[i][j] = computeCost(theta_new,X_exp,y) 
		

fig = plt.figure()
ax = Axes3D(fig)
theta0,theta1 = np.meshgrid(theta0,theta1)
ax.plot_surface(theta0,theta1,J_vals)
fig.show()

fig3 = plt.figure()
ax4 = fig3.add_subplot(111)
ax4.contourf(theta0, theta1, J_vals, 8, alpha=.75, cmap='jet')
fig3.show()


