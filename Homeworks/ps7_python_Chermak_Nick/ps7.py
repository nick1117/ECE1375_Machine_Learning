import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from predict import predict
from sklearn.metrics import accuracy_score
from sigmoidGradient import sigmoidGradient
from sigmoid import sigmoid
from nnCost import nnCost
from sGD import sGD
from sklearn.metrics import accuracy_score
from tabulate import tabulate

#Question 0
#part a
data = scipy.io.loadmat("input\\HW7_Data2_full.mat")
X = data["X"]  
y = data["y_labels"].flatten()  
rand_image = np.random.permutation(X.shape[0])
fig, axes = plt.subplots(4, 4)
for i, ax in enumerate(axes.flat):
    image = X[rand_image[i], :].reshape(32, 32)
    #image = np.reshape(X[rand_image[i], :],(-32,32))
    label = y[rand_image[i]]
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {label}")
plt.tight_layout()
plt.savefig("output\\ps7-0-a-1.png")
plt.show()

#part b
X_train = X[rand_image[:13000]]
y_train = y[rand_image[:13000]]
X_test = X[rand_image[13000:]]
y_test = y[rand_image[13000:]]

#Question 1
#part a
weights = scipy.io.loadmat("input\\HW7_weights_3_full.mat")
Theta1 = weights["Theta1"]  
Theta2 = weights["Theta2"]
input_layer_size = Theta1.shape[1]
hidden_layer_size = Theta1.shape[0]
prediction, h_x = predict(Theta1, Theta2, X)

#part b
accuracy = accuracy_score(y, prediction)
print(f'The accuracy of the prediction from the function is: {accuracy*100} %')


#Question 2
lams = [0.1, 1, 2]
K = np.max(y)
for i, lam in enumerate(lams):
    J = nnCost(Theta1, Theta2, X, y, K, lam)
    print(f'The cost using lambda = {lam} is: {J}')

#Question 3
z = np.array([-10, 0, 10])
g_prime_test = sigmoidGradient(z)
print(f'The sigmoid gradient when z = [-10, 0, 10] is: {g_prime_test}')

#Question 4
alpha = 0.01
Theta1, Theta2 = sGD(input_layer_size,hidden_layer_size,K,X_train,y_train,1,alpha,10)

print(f'I used an alpha value of: {alpha}')

#Question 5
epochs = [50, 300]
#epochs = [2, 3]
print(epochs[0])
headers = ['Lambda', 'Training Data Accuracy', 'Testing Data Accuracy']
total_costs = []
for epoch in epochs:
    train_values = []
    test_values = []
    costs = []
    for lam in lams:
        Theta1, Theta2 = sGD(input_layer_size,hidden_layer_size,K,X_train,y_train,lam,alpha,epoch)

        train_pred, train_h = predict(Theta1.T, Theta2.T, X_train)
        test_pred, test_h = predict(Theta1.T, Theta2.T, X_test)

        accuracy_train = accuracy_score(y_train, train_pred) * 100
        accuracy_test = accuracy_score(y_test, test_pred) * 100

        train_values = np.hstack([train_values,accuracy_train])
        test_values = np.hstack([test_values,accuracy_test])

        cost_train = nnCost(Theta1.T, Theta2.T, X_train, y_train, K, lam)
        costs = np.hstack([costs,cost_train])
        

    data = np.column_stack([lams,train_values,test_values])
    total_costs = np.append(total_costs, costs)
    print(f'Results for {epoch} Max Epoch')
    print(tabulate(data, headers=headers, showindex=True, tablefmt="grid"))

total_costs = total_costs.reshape((len(lams),len(epochs)))
total_costs = np.column_stack([lams, total_costs])
cost_header = ['Lambda', f'Costs for Max Epoch: {epochs[0]}', f'Cost for Max Epoch: {epochs[1]}']
print('Cost Table:')
print(tabulate(total_costs, headers=cost_header, showindex=True, tablefmt="grid"))





        
