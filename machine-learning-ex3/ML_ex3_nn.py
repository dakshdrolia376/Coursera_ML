import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt
import scipy.io as scio

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

def holdOn():
    raw_input ('Press Enter to continue.....\n')

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
def predict(Theta1,Theta2,X):
    m = X.shape[0]
    X = np.concatenate((np.ones((m,1)),X),axis = 1)
    num_labels = Theta2.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        z1 = Theta1.dot(X[i].reshape(X.shape[1],1))
        a1 = sigmoid(z1)
        a1 = np.concatenate(([[1]],a1),axis = 0)
        z2 = Theta2.dot(a1)
        h_theta = sigmoid(z2)
        p[i] = np.argmax(h_theta,axis = 0) + 1
        
    return p
# Loading and visualizing data
print 'Loading and visualizing data....\n'
data = scio.loadmat('ex3/ex3data1.mat')
X = data['X']
y = data['y']
m = y.size
rand_indices = np.random.choice(X.shape[0],20)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1 = plt.imshow(X[rand_indices,:].reshape((400,20)).T,cmap = 'gray_r')
fig1.show() 

holdOn()

# Loading Parameters
print 'Loading Saved Neural Network Parameters'
weights = scio.loadmat('ex3/ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
pred = predict(Theta1,Theta2,X)
print np.mean(np.array(y==pred,dtype = float))*100