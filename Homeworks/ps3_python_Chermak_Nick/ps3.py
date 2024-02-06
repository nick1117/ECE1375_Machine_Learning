import numpy as np
import pandas as pd
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
#Y_train.shape = (len(Y_train),)
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
#Get prediction
h_pred = sigmoid(X_test @ theta_opt)
count = 0 
#if predicition is greater or equal to 0.5 make it a 1 if not make it a 0
#if the prediction at the index matches the Ytest data, increatment count
for i in range (len(h_pred)):
    if h_pred[i] >= 0.5:
        h_pred[i] = 1
    else:
        h_pred[i] = 0

    if h_pred[i] == Y_test[i]:
        count += 1
accuracy = count / len(Y_test)
print(accuracy)

#Part i
#create data for tests with normalized value
scores = np.array([1, 60, 65])
scores.shape = (1,len(scores))
h_scores = sigmoid(scores @ theta_opt)
#text output
print(f'The admission probability is {h_scores}')
if h_scores >= 0.5:
    print('The decision should be to admit')
else:
    print('The decision should be to not admit')



#Question 2
#Part a
data2 = pd.read_csv('input\\hw3_data2.csv', header=None)
X2 = data2.values[:, 0] 
Y2 = data2.values[:, 1] 
X2.shape = (len(X2),1)
Y2.shape = (len(Y2),1)
X2 = np.concatenate((X2,X2**2), axis = 1)
X2 = np.concatenate((np.ones((len(Y2),1)), X2), axis = 1)

#normal equation from HW 2 
theta_2 = np.linalg.pinv(X2.T @ X2) @ X2.T @ Y2
print(f'The optimal values of theta are: {theta_2.T}')

#part b
x_val_2 = np.linspace(np.min(X2[:,1]),np.max(X2[:,1]), num=100)
y_val_2 = theta_2[0] + theta_2[1]*x_val_2 + theta_2[2]*(x_val_2 ** 2)
plt.plot(x_val_2,y_val_2,'b-', label = 'Learned model')
plt.scatter(X2[:,1], Y2[:,0])
plt.xlabel('Population in thouthands, n')
plt.ylabel('profit')
plt.savefig('output\\ps3-2-b.png')
plt.show()

