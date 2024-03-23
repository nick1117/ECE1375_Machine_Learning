import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import accuracy_score

#Question 0
data = pd.read_csv('input\\iris_dataset.csv', header=None)
X = data.values[:, :-1]
Y = data.values[:, -1] 

rows, cols = X.shape
rand_rows = np.random.permutation(rows)
X_train = X[rand_rows[:125]]
Y_train = Y[rand_rows[:125]]

X_test = X[rand_rows[125:]]
Y_test = Y[rand_rows[125:]]

X_train_1 = X_train[Y_train == 1]
X_train_2 = X_train[Y_train == 2]
X_train_3 = X_train[Y_train == 3]

print(f'The shape of X_train_1 is {X_train_1.shape}')
print(f'The shape of X_train_2 is {X_train_2.shape}')
print(f'The shape of X_train_3 is {X_train_3.shape}')

#Question 1
#Part a
mean_values = []
std_values = []
for i in range(cols):
    means = []
    means.append(np.mean(X_train_1[:, i]))
    means.append(np.mean(X_train_2[:, i]))
    means.append(np.mean(X_train_3[:, i]))
    mean_values.append(means)           #rows = feature, col = class ---- 4x3
    stds = []
    stds.append(np.std(X_train_1[:, i]))
    stds.append(np.std(X_train_2[:, i]))
    stds.append(np.std(X_train_3[:, i]))
    std_values.append(stds)  
    
headers = ['Feature #', 'Class 1', 'Class 2', 'Class 3']
print("Mean Values:")
print(tabulate(mean_values, headers=headers, showindex=True, tablefmt="grid"))
print("Standard Deviation Values:")
print(tabulate(std_values, headers=headers, showindex=True, tablefmt="grid"))

#Part b
class_predictions = []
for entry in X_test:
    ln_posterior_classes = []
    for i in range(3): #3 - for 3 classes
        ln_p_sum = 0
        for j in range(cols): #4 - for 4 features
            x = entry[j]
            #mean and std matrix are feature by class so 4 x 3
            p = (1 / (std_values[j][i] * np.sqrt(2 * np.pi))) * np.exp(-((x - mean_values[j][i]) ** 2) / (2 * std_values[j][i] ** 2))
            ln_p = np.log(p)
            ln_p_sum += ln_p
        ln_posterior = ln_p_sum + np.log(1/3)
        ln_posterior_classes.append(ln_posterior)
    predicted_class = np.argmax(ln_posterior_classes) + 1
    class_predictions.append(predicted_class)

#Part c
accuracy = accuracy_score(Y_test, class_predictions)
print(f'The accuracy of the Naive-Bayes classifier is: {accuracy*100} %')

#Question 2
#Part a
Sigma_1 = np.cov(X_train_1.T)
Sigma_2 = np.cov(X_train_2.T)
Sigma_3 = np.cov(X_train_3.T)
print(f'The size of covariance matrix 1: {Sigma_1.shape}')
print(f'The size of covariance matrix 2: {Sigma_2.shape}')
print(f'The size of covariance matrix 3: {Sigma_3.shape}')
print('Class 1 covariance:')
print(Sigma_1)
print('Class 2 covariance:')
print(Sigma_2)
print('Class 3 covariance:')
print(Sigma_3)

#Part b
mean_1 = np.array(mean_values)[:, 0]
mean_2 = np.array(mean_values)[:, 1]
mean_3 = np.array(mean_values)[:, 2]
print(f'The size of mean 1 vector is: {mean_1.shape}')
print(f'The size of mean 2 vector is: {mean_2.shape}')
print(f'The size of mean 3 vector is: {mean_3.shape}')
print('Class 1 mean vector: ')
print(mean_1)
print('Class 2 mean vector: ')
print(mean_2)
print('Class 3 mean vector: ')
print(mean_3)

#Part c
#need or will not work
mean_1 = mean_1.reshape(len(mean_1),1)
mean_2 = mean_2.reshape(len(mean_2),1)
mean_3 = mean_3.reshape(len(mean_3),1)

class_prediction_Q2 = []
for entry in X_test:
    x = entry.reshape(len(entry), 1)
    g1 = -0.5 * (x - mean_1).T @ np.linalg.inv(Sigma_1) @ (x - mean_1) + np.log(1/3) - (len(entry)/2) * np.log(2 * np.pi) - (0.5) * np.log(np.linalg.det(Sigma_1))
    g2 = -0.5 * (x - mean_2).T @ np.linalg.inv(Sigma_2) @ (x - mean_2) + np.log(1/3) - (len(entry)/2) * np.log(2 * np.pi) - (0.5) * np.log(np.linalg.det(Sigma_2))
    g3 = -0.5 * (x - mean_3).T @ np.linalg.inv(Sigma_3) @ (x - mean_3) + np.log(1/3) - (len(entry)/2) * np.log(2 * np.pi) - (0.5) * np.log(np.linalg.det(Sigma_3))
    g = [g1, g2, g3]
    #print(g)
    predicted_class = np.argmax(g) + 1
    class_prediction_Q2.append(predicted_class)
    
accuracy_Q2 = accuracy_score(Y_test, class_prediction_Q2)
print(f'The accuracy of using MLE and Discriminant function for classification is: {accuracy_Q2 * 100} %')
print(f'The accuracy of the naïve classifier and the MLE based classifier seem to both be high (88% and above) however no method seems to always perform better')