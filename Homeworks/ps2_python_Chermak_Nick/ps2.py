import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn


# Question 1 - Cost Function

# X matrix values with bias
X = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4]])
# y values
y = np.array([8, 6, 4, 2])
y.shape = (4,1)

# theta values
theta1 = np.array([0, 1, 0.5])
theta2 = np.array([10, -1, -1])
#theta2.shape=(3,1)
theta3 = np.array([3.5, 0, 0])
#theta4 = np.array([10, -1.077211, -0.922788])
#compute cost values
cost1 = computeCost(X, y, theta1)
cost2 = computeCost(X, y, theta2)
cost3 = computeCost(X, y, theta3)
#cost4 = computeCost(X,y,theta4)
#Text output of test cases
print(f'cost for test case 1: {cost1}')
print(f'cost for test case 2: {cost2}')
print(f'cost for test case 3: {cost3}')



# Question 2 - Gradient descent
#define variables
alpha = 0.01
iters = 15
#call gradient descent
theta, cost_history = gradientDescent(X,y,alpha,iters)
#text output
print(f'The thetas have the respective values of: {theta}')
print(f'The cost after {iters} iterations is: {cost_history[iters-1]}')


# Question 3 - nomral Eqn
theta_NE = normalEqn(X,y)
print(theta_NE)

# Question 4 Linear regression with one variable
#Part a
#read data from file
data1 = pd.read_csv('input\\hw2_data1.csv', header=None, names=['Horsepower', 'Price'])

#Part b
#create scatter plot with labels
plt.scatter(data1['Horsepower'], data1['Price'], marker = "x")
plt.title('Car prices vs horsepower')
plt.xlabel('horsepower (100s hp)')
plt.ylabel('Price (in $1,000)')
plt.savefig('output\\ps2-4-b.png')
plt.show()

#Part c
##create X and Y matrices with the input data 
X = np.concatenate([np.ones((data1.shape[0], 1)), data1['Horsepower'].values.reshape(-1, 1)], axis=1)
Y = data1['Price'].values.reshape(-1, 1) #rehape changes it from 1xn to nx1
#print size of X and Y
print(f'Size of X for question 4 is : {X.shape}')
print(f'Size of Y for question 4 is : {Y.shape}')

#Part d
#radnomize rows 
col, rows = X.shape
rand_rows = np.random.permutation(col)
#create training and testing data sets
X_train = X[rand_rows[:int(col*0.9)]]
X_test = X[rand_rows[int(col*0.9):]]
Y_train = Y[rand_rows[:int(col*0.9)]]
Y_test = Y[rand_rows[int(col*0.9):]]

#Part e
#define variables
alpha = 0.3
iters = 500
#call gradient descent
theta, cost_history4 = gradientDescent(X_train,Y_train,alpha,iters)
#plot cost function vs iterations
plt.plot(cost_history4)
plt.title('Cost vs iterations')
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.savefig('output\\ps2-4-e.png')
plt.show()
#Text output
print(f'The theta values for question 4 part e are: {theta.T}')

#Part f
#create linspace between min and max of X
x_val = np.linspace(np.min(X[:,1]),np.max(X[:,1]), num=100)
#use that to create a line
y_val = x_val * theta[1] + theta[0]
#plot line
plt.plot(x_val, y_val, 'b--', label = 'learned model')
#plot scatter data
plt.scatter(X_train[:,1], Y_train, marker = 'x', label = 'training data')
plt.title('Car prices vs horsepower')
plt.xlabel('horsepower (100s hp)')
plt.ylabel('Price (in $1,000)')
plt.savefig('output\\ps2-4-f.png')
plt.show()

#Part g
y_pred = X_test @ theta
#error_pred1 =  np.mean((np.sum((y_pred - Y_test)**2)))
error_pred = 2 * computeCost(X_test,Y_test,theta) #multiply by 2 since cost function has a 1/2
#print(error_pred)
#print(error_pred1)
print(f'The prediction error in part g using gradientDescent is: {error_pred}')

#Part h
#find prediction error using normal equation
theta_h = normalEqn(X_train, Y_train)
Y_pred_h = X_test @ theta_h
error_pred_h = 2 * computeCost(X_test,Y_test,theta_h)
#error_pred_h = np.mean(np.sum((y_pred - y_test)**2))
print(f'The prediction error in part h using normalEqn is: {error_pred_h}')

#Part i
iters = 300
alpha = [0.001, 0.003, 0.03, 3]
#plot cost function for each alpha
for x in range(len(alpha)):
    theta, cost_history = gradientDescent(X_train, Y_train, alpha[x], iters)
    plt.plot(cost_history)
    plt.title(f'Cost using {x} alpha vs iterations')
    plt.xlabel('iterations')
    plt.ylabel('Cost')
    plt.savefig(f'output\\ps2-4-i-{x}.png')
    plt.show()


#Question 5

#Part a
#read data and set to X and Y
data2 = pd.read_csv('input\\hw2_data3.csv', header=None)
X2 = data2.values[:, :-1] 
Y2 = data2.values[:, -1]  
Y2.shape = (len(Y2),1)
#standardize data
Xmean = np.mean(X2, axis=0)
Xstd = np.std(X2, axis=0)
X2norm = (X2 - Xmean) / Xstd
#normalize
ones = np.ones((len(Y2),1))
X2norm = np.concatenate((ones, X2norm), axis = 1)
#mean and std of normalized data
X2normMean = np.mean(X2norm, axis=0)
X2normStd = np.std(X2norm, axis=0)
#text output
print(f'The mean and std of the first feature are: {Xmean[0]} and {Xstd[0]}')
print(f'The mean and std of the first normalized feature are: {X2normMean[1]} and {X2normStd[1]}') #added 1 vector so next index (want to make sure its normalized)
print(f'The mean and std of the second feature are: {Xmean[1]} and {Xstd[1]}')
print(f'The mean and std of the second normalized feature are: {X2normMean[2]} and {X2normStd[2]}') #added 1 vector
print(f'The size of the feature matrix X is: {X2norm.shape}')
print(f'The size of the label vector y is: {Y2.shape}')

#Part b
#set values
alpha5 = 0.01
iter5 = 750
#call function
theta5, cost_history5 = gradientDescent(X2norm, Y2, alpha5, iter5)
#plot
plt.plot(cost_history5)
plt.title(f'Cost vs iterations: 5b')
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.savefig(f'output\\ps2-5-b.png')
plt.show()
#text output
print(f'The values of theta are: {theta5.T}')

#Part c
#calculate prediction by inputing the new data 
new_data = np.array([2300, 1300])
new_data.shape = (1,len(new_data))
NDnorm = (new_data - Xmean) / Xstd #use mean and std from before
ones = np.ones((1,1))
NDnorm = np.concatenate((ones, NDnorm), axis = 1)
h = NDnorm @ theta5 #equation for h
print(f'Prediction CO2 emission of a car whose engine size is 2300 and weight is 1300 is: {h}')