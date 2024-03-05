import numpy as np
import scipy.io
from tabulate import tabulate
from weightedKNN import weightedKNN
from image_sort import image_sort
import os as os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import time

#Question 1:
#Part a)
#Load Data and set to X's and y's
data3 = scipy.io.loadmat("input\\hw4_data3.mat")
X_test = np.array(data3["X_test"])
X_train = np.array(data3["X_train"])
y_test = np.array(data3["y_test"])
y_train = np.array(data3["y_train"])
#X_test = np.concatenate((np.ones((len(y_test),1)), X_test), axis = 1)
#X_train = np.concatenate((np.ones((len(y_train),1)), X_train), axis = 1)
#y_pred = weightedKNN(X_train, y_train, X_test, 0.01)

#part b)
#create table of sigmas and create accuracy vector to be populated
sigmas = [0.01, 0.07, 0.15, 1.5, 3, 4.5]
accuracy = []
#loop over sigmax
for i, sigma in enumerate(sigmas):
    #get y predictions
    y_pred = weightedKNN(X_train, y_train, X_test, sigma)
    #find how many times they match
    count = 0                              
    for i in range (len(y_test)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    #compute accuracy and append to accuracy vector
    acc = count / len(y_test)
    accuracy.append(acc)
#create data and plot it
data = [sigmas, accuracy]
print(tabulate([data], headers=["Sigma", "Accuracy"], tablefmt="grid"))
print("the best accuracy seems to come when sigma is between 0.07 adn 0.15 with a steep drop off before 0.07 and a more gradual decline when sigma is heiger then 0.15")

#Question 2
#part 0
image_sort()
rand_subject = np.random.permutation(320) + 1
train_list = os.listdir(f"input\\train")
fig, axes = plt.subplots(1, 3)
for i in range(3):
    img_path = os.path.join("input\\train", train_list[rand_subject[i]])
    image = mpimg.imread(img_path)
    axes[i].imshow(image,cmap="gray")
    name = os.path.splitext(train_list[rand_subject[i]])[0]
    personID, imageID = name.split("_")
    axes[i].set_ylabel(f"PersonID: {personID}")
    axes[i].set_xlabel(f"ImageID: {imageID}")
fig.tight_layout()
plt.savefig("output\\ps5-2-0.png")
plt.show()

#part a
# initialize image matrix with proper size
images_matrix = np.zeros((10304, len(train_list)))

#loop through file path and select each file name
for i, file_name in enumerate(train_list):
    image_path = os.path.join(f"input\\train", file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    images_matrix[:, i] = image.flatten()

plt.imshow(images_matrix, cmap="gray")
plt.show()
plt.savefig("output\\ps5-1-a.png")

#part b
avg_face_vect = np.mean(images_matrix,axis=1)
avg_face = np.reshape(avg_face_vect,(112,92))
plt.imshow(avg_face, cmap = "gray")
plt.show()
plt.savefig("output\\ps5-2-1-b.png")
print('The face looks very creepy, like slenderman.')

#part c
avg_face_vect.shape = (len(avg_face_vect),1)
A = images_matrix - avg_face_vect
C = A @ A.T
plt.imshow(C, cmap = "gray")
plt.show()
plt.savefig("output\\ps5-2-1-c.png")

#part d
#matrix_eig = A.T @ A
eigenvalues, eigenvectors = LA.eig(A.T @ A)

sorted_index = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[sorted_index]
eigenvectors = eigenvectors[:,sorted_index]

total_sum = np.sum(eigenvalues)
k_sum = 0
percent_variance_list = []
K = 0
k_list = []
for i in eigenvalues:
    k_sum += i
    percent_variance = k_sum / total_sum
    percent_variance_list.append(percent_variance)
    K += 1
    k_list.append(K)
    if percent_variance >= 0.95:
        break

plt.plot(k_list,percent_variance_list)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('K')
plt.ylabel('Percent Variance preserved')
plt.savefig('output\\ps5-2-1-d.png')
plt.show()
print(f"Number of eigenvectors for at least 95% of the variance: {K}")

#Part e
eigvals, eigvecs = eigs(C,k=K)
U = eigvecs.real
fig1, axes1 = plt.subplots(1, 9)
for i in range(9):
    eigenface = np.reshape(U[:,i],(112,92))
    axes1[i].imshow(eigenface,cmap="gray")
    axes1[i].axis('off')
fig1.tight_layout()
plt.savefig("output\\ps5-2-1-e.png")
plt.show()

#Question 2.2
#part a
#mean vector = avg_face_vector, U_i is ith column of U
avg_face_vect.shape = (len(avg_face_vect),1)
w_train = []
y_train = []
for i, file_name in enumerate(train_list):
    image_path = os.path.join(f"input\\train", file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    I = image.flatten()
    I.shape = (len(I),1)
    
    w = U.T @ (I - avg_face_vect)
    w_train.append(w)

    personID, imageID = file_name.split("_")
    y_train.append(personID)

w_train = np.array(w_train)
y_train = np.array(y_train)
w_train = w_train.reshape((i+1,K))
y_train = y_train.reshape((i+1,1))

#part b
w_test = []
y_test = []
test_list = os.listdir(f"input\\test")
for i, file_name in enumerate(test_list):
    image_path = os.path.join(f"input\\test", file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    I = image.flatten()
    I.shape = (len(I),1)
    w = U.T @ (I - avg_face_vect)
    w_test.append(w)

    personID, imageID = file_name.split("_")
    y_test.append(personID)

w_test = np.array(w_test)
y_test = np.array(y_test)
w_test = w_test.reshape((i+1,K))
#y_test = y_test.reshape((i+1,1))


print(f"The dimensions fo w_train are: {np.shape(w_train)}")
print(f"The dimensions fo w_test are: {np.shape(w_test)}")

#question 2.3
#part a

K_values = [1, 3, 5, 7, 9, 11]
accuracy_data = []
y_total = len(y_test)

y_train = y_train.ravel()
for index, k in enumerate(K_values):
    knn = KNeighborsClassifier(k)
    knn.fit(w_train,y_train)
    prediction = knn.predict(w_test)
    #prediction = prediction.ravel()
    correct = sum(prediction == y_test)
    accuracy = correct / y_total
    #accuracy_data[index] = accuracy
    accuracy_data.append(accuracy)
data1 = [K_values, accuracy_data]
print(tabulate([data1], headers=["K", "Accuracy"], tablefmt="grid"))
print("The accuracy is highest when K is lower and drops off as K increases.")

#part b
#SVM kernel{‘linear’, ‘poly’, ‘rbf’} , decision_function_shape {‘ovo’, ‘ovr’}

kernels = ['linear', 'poly', 'rbf']
decision_shapes = ['ovo','ovr']
train_times = []
accuracies = []
for kernel in kernels:
    i = 1
    for decision_shape in decision_shapes:
        clf = SVC(kernel=kernel,decision_function_shape=decision_shape)
        if decision_shape == 'ovr':
            clf = OneVsRestClassifier(SVC(kernel=kernel))
        start = time.time()
        clf.fit(w_train, y_train)
        end = time.time()
        total_time = end - start
        if i == 1:
            total_time_ovo = total_time
        elif i == 2:
            total_time_ovr = total_time
            train_times.append((kernel, total_time_ovo, total_time_ovr))
        y_pred = clf.predict(w_test)
        correct = sum(y_pred == y_test)
        accuracy = correct / y_total
        if i == 1:
            accuracy_ovo = accuracy
        elif i == 2:
            accuracy_ovr = accuracy
            accuracies.append((kernel, accuracy_ovo, accuracy_ovr))
        i += 1


print("Table for training times for each model")
print(tabulate(train_times, headers=["Decision Function Shape", "one vs one", "one vs all"], tablefmt="grid"))

print("Table for accuracies for each model")
print(tabulate(accuracies, headers=["Decision Function Shape", "one vs one", "one vs all"], tablefmt="grid"))