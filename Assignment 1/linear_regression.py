import numpy as np
from matplotlib import pyplot
import math
from mpl_toolkits.mplot3d import axes3d


# Load the data
data = np.loadtxt('linear_regression.txt', delimiter = ',')
#separate predictor from target variable
X = np.c_[np.ones(data.shape[0]), data[:,0]]
y = np.c_[data[:,1]]


# First appraoch - Normal equation

def normalEquation(X,y):
    X.T
    xTx = X.T.dot(X)  
    XtX = np.linalg.inv(xTx)
    XtX_xT = XtX.dot(X.T)
    theta =XtX_xT.dot(y)
    return theta
    """
    Parameteres: input variables (Table) , Target vector
    Instructions: Complete the code to compute the closed form solution to linear regression and 	save the result in theta.
    Return: coefficinets 
    """
    ## Your codes go here 
  
    

print (theta)

# Iterative Approach - Gradient Descent 

'''
Following paramteres need to be set by you - you may need to run your code multiple times to find the best combination 
'''

"""
    Paramters: input variable , Target variable, theta, number of iteration, learning_rate
    Instructions: Complete the code to compute the iterative solution to linear regression, in each iteration you will 
    add the cost of the iteration to a an empty list name cost_hisotry and update the theta.
    Return: theta, cost_history 
    """
    
    # Your code goes here 
    
    # print(past_theta[0])
theta = np.zeros([2,1]) 
iterations = 1000
alpha = 0.01
m=len(y)
#X = np.hstack((ones, X)) # adding the intercept term
cost_history = []
theta_history = []

def cal_cost(theta,X,y):
    m=len(y)
    predictions=X.dot(theta)
    cost=(1/2*m)*np.sum(np.square(predictions-y))
    return cost


def gradient_descent(X,y,theta,alpha,iterations):
    m=len(y)
    cost_history=np.zeros(iterations)
    theta_history=np.zeros((iterations,2))
    for it in range(iterations):
        prediction=np.dot(X,theta)
        
        theta=theta-(1/m)*alpha*(X.T.dot(prediction-y))
        theta_history[it,:]=theta.T
        cost_history[it]=cal_cost(theta,X,y)
        
    return theta,cost_history,theta_history

theta,cost_history,theta_history=gradient_descent(X,y,theta,alpha,iterations)
# Plot the cost over number of iterations
'''
Your plot should be similar to the provided plot
'''
pyplot.title('Cost Function')
pyplot.xlabel('No. of iteration')
pyplot.ylabel('Cost function')
pyplot.plot(cost_history)
pyplot.show()



# Plot the linear regression line for both gradient approach and normal equation in same plot
'''
hints: your x-axis will be your predictor variable and y-axis will be your target variable. plot a
scatter plot and draw the regression line using the theta calculated from both approaches. Your plot
should be similar to what provided.
'''



# Plot contour plot and 3d plot for the gradient descent approach

import matplotlib.pyplot as plt
plt.figure()
cp = plt.contour(np.transpose(theta_history))
plt.ylim(-1,3)
plt.xlim(100,400)


plt.title('Filled Contours Plot')
plt.xlabel(' theta 0')
plt.ylabel('theta 1')
plt.show()
'''
your plots should be similar to our plots.



'''
fig = plt.figure(theta_history)
ax = fig.add_subplot(111, projection='3d')
ax = axes3D(fig)     
axes3d.plot_surface(theta_history)










