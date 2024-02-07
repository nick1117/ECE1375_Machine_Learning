import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from Reg_normalEqn import Reg_normalEqn
from computeCost import computeCost
from logReg_multi import logReg_multi



##Load Data
data1 = scipy.io.loadmat('input\\hw4_data1.mat')
data2 = scipy.io.loadmat('input\\hw4_data2.mat')
data3 = scipy.io.loadmat('input\\hw4_data3.mat')

#Set data 1
#print(data1.keys())
X_data = np.array(data1["X_data"])
y = np.array(data1["y"])

#set data 2
#print(data2.keys())
X1 = np.array(data2["X1"])
X2 = np.array(data2["X2"])
X3 = np.array(data2["X3"])
X4 = np.array(data2["X4"])
X5 = np.array(data2["X5"])
y1 = np.array(data2["y1"])
y2 = np.array(data2["y2"])
y3 = np.array(data2["y3"])
y4 = np.array(data2["y4"])
y5 = np.array(data2["y5"])

#Set Data 3
#print(data3.keys())
X_test = np.array(data3["X_test"])
X_train = np.array(data3["X_train"])
y_test = np.array(data3["y_test"])
y_train = np.array(data3["y_train"])


#Question 1
#Part a - build Reg_normalEqn function

#Part b
#add offset feature
X_data = np.concatenate((np.ones((len(y),1)), X_data), axis = 1)
#Text output
print(f'The size of the feature matrix is {X_data.shape}')

#part c
#Declare variables like how many iterations, matrix of lambdas and initiallizing the matrix that will be 20x8
iters = 20
L = [0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017]
error_train_total = []
error_test_total = []
#start loop
for i in range (iters):
    #creat random test and train from data using 88%
    rows, cols = y.shape
    rand_rows = np.random.permutation(rows)
    X_train_1c = X_data[rand_rows[:int(rows*0.88)]]
    X_test_1c = X_data[rand_rows[int(rows*0.88):]]
    y_train_1c = y[rand_rows[:int(rows*0.88)]]
    y_test_1c = y[rand_rows[int(rows*0.88):]]
    
    #initialize vectors for error over lambda for this iteration and clear it from last iteration
    error_train_vect = []
    error_test_vect = []

    for j in L:
        #calculate theta
        theta = Reg_normalEqn(X_train_1c, y_train_1c, j)
        #calculate error
        error_train = computeCost(X_train_1c, y_train_1c, theta)
        error_test = computeCost(X_test_1c, y_test_1c, theta)
        #add to vector initialized for this iteration
        error_train_vect.append(error_train)
        error_test_vect.append(error_test)

    #add to total error vector declared befroe loop
    error_train_total.append(error_train_vect)
    error_test_total.append(error_test_vect)

#compute mean of each column
col_avg_train = np.mean(error_train_total, axis = 0)
col_avg_test = np.mean(error_test_total, axis = 0)
#plot data
plt.plot(L,col_avg_train, marker = '*', color = 'r')
plt.plot(L,col_avg_test, marker = 'o', color = 'b', mfc='none')
plt.xlabel('Lambda')
plt.ylabel('average error')
plt.legend(["training error", "testing error"], loc="upper right")
plt.savefig('output\\ps4-1-a.png')
plt.show()
#Text output - first find the minimum value of the test error and use that indxe
min_lamdba_test_index = np.argmin(col_avg_test)
print(f'The suggested lambda is: {L[min_lamdba_test_index]}')

#Question 2
#load 1 set
X_train1 = np.concatenate([X1, X2, X3, X4], axis=0)
X_test1 = X5
y_train1 = np.concatenate([y1, y2, y3, y4], axis=0)
y_test1 = y5
#load 2 set
X_train2 = np.concatenate([X1, X2, X3, X5], axis=0)
X_test2 = X4
y_train2 = np.concatenate([y1, y2, y3, y5], axis=0)
y_test2 = y4
#load 3 set
X_train3 = np.concatenate([X1, X2, X4, X5], axis=0)
X_test3 = X3
y_train3 = np.concatenate([y1, y2, y4, y5], axis=0)
y_test3 = y3
#load 4 set
X_train4 = np.concatenate([X1, X3, X4, X5], axis=0)
X_test4 = X2
y_train4 = np.concatenate([y1, y3, y4, y5], axis=0)
y_test4 = y2
#load 5 set
X_train5 = np.concatenate([X2, X3, X4, X5], axis=0)
X_test5 = X1
y_train5 = np.concatenate([y2, y3, y4, y5], axis=0)
y_test5 = y1

#part a
#for K in range(1,15,2):
#    for num in range(1, 6):
#        X_train_current = X_train+str(num)
#        y_train_current = y_train+str(num)

#Question 3

