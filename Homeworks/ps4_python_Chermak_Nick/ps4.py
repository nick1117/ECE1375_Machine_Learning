import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Reg_normalEqn import Reg_normalEqn
from computeCost import computeCost
from logReg_multi import logReg_multi
from sklearn.neighbors import KNeighborsClassifier



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
X_data_2 = np.array([data2["X1"], data2["X2"], data2["X3"], data2["X4"], data2["X5"]])
y_data_2 = np.array([data2["y1"], data2["y2"], data2["y3"], data2["y4"], data2["y5"]])

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
#c
#np.random.seed(0)
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
#import pdb; pdb.set_trace()
#compute mean of each column once the whole matrix is made
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

#part a
K = [1, 3, 5, 7, 9, 11, 13, 15]
accuracy = np.zeros((len(X_data_2),len(K)))
for i, features in enumerate(X_data_2):
    Current_Xtrain = np.delete(X_data_2, i, axis=0)
    Current_Xtrain = np.vstack(Current_Xtrain)
    Current_yTrain = np.delete(y_data_2, i, axis = 0)
    Current_yTrain = np.vstack(Current_yTrain)
    Current_yTrain = Current_yTrain.ravel()
    Current_yTest = y_data_2[i].ravel()
    for j, k in enumerate(K):
        knn = KNeighborsClassifier(k)
        knn.fit(Current_Xtrain,Current_yTrain)
        prediction = knn.predict(features)
        prediction = prediction.ravel()
        accuracy[i][j] = np.average(prediction == Current_yTest)
avg_acc = np.average(accuracy, axis=0)

plt.plot(K,avg_acc)
plt.xlabel('K')
plt.ylabel('Average Accuracy')
plt.xticks(K)
plt.savefig('output\\ps4-2-a.png')
plt.show()

#Question 3
y_test_predict = logReg_multi(X_train, y_train, X_test)
y_train_predict = logReg_multi(X_train, y_train, X_train)
test_count = 0
train_count = 0                              
for i in range (len(y_test)):
    if y_test_predict[i] == y_test[i]:
        test_count = test_count + 1
for i in range (len(y_train)):
    if y_train_predict[i] == y_train[i]:
        train_count = train_count + 1

accuracy_train = train_count / len(y_train)
accuracy_test = test_count / len(y_test)


data = {'Accuracy': [accuracy_train, accuracy_test]}
labels = ['training', 'testing']
acc_table = pd.DataFrame(data)
print(acc_table)
#df = pd.DataFrame(acc_table)
#print(df)