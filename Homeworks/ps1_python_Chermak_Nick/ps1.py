import numpy as np # linear algebra
import matplotlib.pyplot as plt
import time

#Problem 3

#Part a
X = np.random.normal(1.5,0.6,1000000);

#Part B
Z = np.random.uniform(-1,3,1000000);

#Part C
plt.figure()
plt.hist(X)
plt.savefig('c:\\Users\\nicka\\CS\\ECE_1375_ML\\Homeworks\\ps1_python_Chermak_Nick\\output\\ps1-3-c-1.png')
plt.figure()
plt.hist(Z)
plt.savefig('c:\\Users\\nicka\\CS\\ECE_1375_ML\\Homeworks\\ps1_python_Chermak_Nick\\output\\ps1-3-c-2.png')
print("X looks like Gausian Distribution and Z looks like unifrom distribution")

#Part D
start = time.time()
X_add = X
for i in range(0,X.shape[0]):
  X_add[i] = X[i] + 1
end = time.time()
print("The execution time with a loop for adding 1 to every element in X is: " + str(end-start) + " seconds")

#Part E
start = time.time()
X = X + 1
end = time.time()
print("The execution time without a loop for adding 1 to every element in X is: " + str(end-start) + " seconds")
print("Seems that not using a loop is more efficent to add a constant")

#Part F
Y = Z[(Z < 1.5) & (Z > 0)]
print("There are: " + str(len(Y)) + " elements in Y")
print("There is a difference between the numbers each time the code is run since Z is randomly generated each time the code is run")

#Problem 4

#Part A
A = np.array([[2,1,3],[2,6,8],[6,8,18]])
min_col1 = A[:,0].min()
min_col2 = A[:,1].min()
min_col3 = A[:,2].min()
min_row1 = A[0,:].min()
min_row2 = A[1,:].min()
min_row3 = A[2,:].min()
sum_col1 = A[:,0].sum()
sum_col2 = A[:,1].sum()
sum_col3 = A[:,2].sum()
sum_total = np.sum(A)
B = np.square(A)

#Part B
C = np.array([1,3,5])
x = np.linalg.solve(A, C)
print(" the solution for x, y, z respectively for the system of equations given in the homework is: " + str(np.transpose(x)))

#Part C
x1 = np.array([-0.5, 0, 1.5])
x2 = np.array([-1, -1, 0])
x1_L1_norm = np.linalg.norm(x1, ord = 1)
x2_L1_norm = np.linalg.norm(x2, ord = 1)
x1_L2_norm = np.linalg.norm(x1, ord = 2)
x2_L2_norm = np.linalg.norm(x2, ord = 2)

print("Calcualted by hand X1_L1 is: " + str(2))
print("Code output for X1_L1 is: " + str(x1_L1_norm))
print("Calcualted by hand X2_L1 is: " + str(2))
print("Code output for X2_L1 is: " + str(x2_L1_norm))
print("Calcualted by hand X1_L2 is: " + str(np.sqrt(2.5)))
print("Code output for X1_L2 is: " + str(x1_L2_norm))
print("Calcualted by hand X2_L2 is: " + str(np.sqrt(2)))
print("Code output for X2_L2 is: " + str(x2_L2_norm))

#Problem 5

#Part A
ones = np.ones((3,10))
count = np.arange(1,11)
X = np.transpose(ones * count)
Y = count.reshape(10,1)
print("Matrix X: \n" + str(X))

#Part B
rand_rows = np.random.permutation(10)
X_train = X[rand_rows[:8]]
X_test = X[rand_rows[8:]]

#Part C
y_train = Y[rand_rows[:8]]
y_test = Y[rand_rows[8:]]

#Part D
print("Matrix X_train: \n" + str(X_train))
print("Matrix X_test: \n" + str(X_test))
print("Matrix y_train: \n" + str(y_train))
print("Matrix y_test: \n" + str(y_test))