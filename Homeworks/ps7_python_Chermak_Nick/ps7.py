import scipy.io
import numpy as np
import matplotlib.pyplot as plt

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

