import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from predict import predict
from sklearn.metrics import accuracy_score
from sigmoidGradient import sigmoidGradient
from sigmoid import sigmoid
from nnCost import nnCost


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
prediction, h_x = predict(Theta1, Theta2, X)

#part b
accuracy = accuracy_score(y, prediction)
print(f'The accuracy of the prediction from the function is: {accuracy*100} %')


#Question 2
lams = [0, 0.1, 1, 2]
K = np.max(y)
for i, lam in enumerate(lams):
    J = nnCost(Theta1, Theta2, X, y, K, lam)
    print(f'The cost using lambda = {lam} is: {J}')

#Question 3
z = np.array([-10, 0, 10])
g_prime_test = sigmoidGradient(z)
print(f'The sigmoid gradient when z = [-10, 0, 10] is: {g_prime_test}')

#Question 4
