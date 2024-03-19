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
mean_std = {'Class 1': {'mean': [], 'std': []},
            'Class 2': {'mean': [], 'std': []},
            'Class 3': {'mean': [], 'std': []}}

for i in range(cols):
    mean_std['Class 1']['mean'].append(np.mean(X_train_1[:, i]))
    mean_std['Class 1']['std'].append(np.std(X_train_1[:, i]))
    mean_std['Class 2']['mean'].append(np.mean(X_train_2[:, i]))
    mean_std['Class 2']['std'].append(np.std(X_train_2[:, i]))
    mean_std['Class 3']['mean'].append(np.mean(X_train_3[:, i]))
    mean_std['Class 3']['std'].append(np.std(X_train_3[:, i]))

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
print(f'The accuracy of the classifier is: {accuracy*100} %')

#Question 2