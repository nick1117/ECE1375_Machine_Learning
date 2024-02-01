import numpy as np
import matplotlib.pyplot as plt
from sigmoid import sigmoid
from costFunction import costFunction
from gradFunction import gradFunction
from scipy.optimize import fmin_bfgs


#Question 1:
#Part a
#load data and make first 2 columns X and last column Y
data = np.loadtxt('input\\hw3_data1.txt', delimiter=',')
X = data[:, :-1] 
Y = data[:, -1]
Y.shape = (len(Y),1) #reshape Y since the matrix has a null shape variable
#Add column of ones to X to normalize data
X = np.concatenate((np.ones((len(Y),1)), X), axis = 1)
print(f'The size of X is: {X.shape}')
print(f'The size of Y is: {Y.shape}')

#Part b
#plot the data
plt.scatter(X[:,1], X[:,2], c=Y, cmap = 'cividis')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
#plt.legend()
plt.savefig('output\\ps3-1-b.png')
plt.show()

#Part C
#randomize data for train and test data by randomizing the rows and then picking those
rows, cols = Y.shape
rand_rows = np.random.permutation(rows)
X_train = X[rand_rows[:int(rows*0.9)]]
X_test = X[rand_rows[int(rows*0.9):]]
Y_train = Y[rand_rows[:int(rows*0.9)]]
Y_test = Y[rand_rows[int(rows*0.9):]]
#print(Y_train.shape) #verify that they were the right shape
#print(Y_test.shape) #Verify that they were the right shape

#Part D
#implement and call sigmoid
z = np.arange(-15, 15.01, 0.01)
#print(z)
#plot data
gz = sigmoid(z)
plt.plot(z,gz)
plt.xlabel('z')
plt.ylabel('gz')
plt.grid(True)
plt.savefig('output\\ps3-1-c.png')
plt.show()
#text response
print('The value the output reaches 0.1 is -ln(9)')

#Part e
#declare, format and normalize toy data
X_toy = np.array([[1,0],[1,3],[3,1],[3,4]])
Y_toy = np.array([[0],[1],[0],[1]])
X_toy = np.concatenate((np.ones((len(Y_toy),1)), X_toy), axis = 1)
theta_toy = np.array([2,0,0])
theta_toy.shape = (len(theta_toy),1)
#call functions for toy data
J_toy = costFunction(theta_toy, X_toy, Y_toy)
Grad_toy = gradFunction(theta_toy, X_toy, Y_toy)
#text output for cost function of toy data
print(f'The cost function using the toy data is: {J_toy}')

#Part f
#declare initial thetas as zero
theta_zeros = np.zeros(X_train.shape[1])
theta_zeros.shape = (len(theta_zeros),1)
#call fmin_bfgs function for convergence
theta_opt = fmin_bfgs(costFunction,theta_zeros, fprime = gradFunction, args = (X_train, Y_train))
theta_opt.shape = (len(theta_opt),1)
cost_opt = costFunction(theta_opt, X_train,Y_train)
#Text outputs of optimal value and final cost function
print(f'The optimal values of theta are: {theta_opt.T}')
print(f'With the optimal thetas, the cost is: {cost_opt}')

#Part g
#Develop equation
x_val = np.linspace(np.min(X_train[:,1]),np.max(X_train[:,1]), num=100)
#equation is: theta0 + theta1*x1 + theta2*x2 >= 0, so solve for x2
x2 = (-(theta_opt[0] + (theta_opt[1] * x_val))) / theta_opt[2]
#plot data
plt.plot(x_val,x2,'b-', label = 'Decision Boundary')
plt.scatter(X[:,1], X[:,2], c=Y, cmap = 'cividis')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.savefig('output\\ps3-1-f.png')
plt.show()

#Part h
accuracy = / len(Y_test)