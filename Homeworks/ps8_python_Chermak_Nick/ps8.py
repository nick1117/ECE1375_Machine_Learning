import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from tabulate import tabulate

data = scipy.io.loadmat("input\\HW8_data1.mat")
X = np.array(data["X"])
y  = np.array(data["y"])
y[y == 10] = 0
y = y.ravel()

#Part a
fig, axes = plt.subplots(5, 5)
rand_nums = np.random.permutation(X.shape[0]) 
for i, ax in enumerate(axes.flat):
    image = np.reshape(X[rand_nums[i]],(20,20))
    ax.imshow(image,cmap="gray")
    ax.axis('off')
    ax.set_title(f'Label: {y[rand_nums[i]]}')
plt.tight_layout()
plt.show()

#Part b
X_train = X[rand_nums[:4500]]
y_train = y[rand_nums[:4500]]
X_test = X[rand_nums[4500:]]
y_test = y[rand_nums[4500:]]

#Part c
subsets = 5
temp_subsets = {}
for i in range(subsets):
    start_index = int(i * (X_train.shape[0]/subsets))
    end_index = int((i + 1) * (X_train.shape[0]/subsets))
    X_subset = X_train[start_index:end_index]
    y_subset = y_train[start_index:end_index]
    temp_subsets[f'X{i+1}'] = X_subset
    temp_subsets[f'y{i+1}'] = y_subset
    
    # Save to matlab
    subset_data = {'X_subset': X_subset, 'y_subset': y_subset}
    scipy.io.savemat(f'input\\subset_{i+1}.mat', subset_data)
    
X1 = temp_subsets['X1']
X2 = temp_subsets['X2']
X3 = temp_subsets['X3']
X4 = temp_subsets['X4']
X5 = temp_subsets['X5']
y1 = temp_subsets['y1']
y2 = temp_subsets['y2']
y3 = temp_subsets['y3']
y4 = temp_subsets['y4']
y5 = temp_subsets['y5']

#Part d

#error
def error(X,y,model):
    model_predictions = model.predict(X)
    model_accuracy = accuracy_score(y, model_predictions)
    error = 1 - model_accuracy
    return error

svm = SVC(kernel='rbf')
svm.fit(X1,y1)

def svm_error(X,y,svm):
    svm_predictions = svm.predict(X)
    svm_accuracy = accuracy_score(y, svm_predictions)
    error = 1 - svm_accuracy
    return error

svm_error1 = error(X1,y1,svm)

svm_error2 = error(X2,y2,svm)
svm_error3 = error(X3,y3,svm)
svm_error4 = error(X4,y4,svm)
svm_error5 = error(X5,y5,svm)
svm_error_test = error(X_test,y_test,svm)

svm_data = np.vstack((svm_error1,svm_error2,svm_error3,svm_error4,svm_error5,svm_error_test))

#Part e
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X2,y2)

def KNN_error(X,y,KNN):
    KNN_predictions = KNN.predict(X)
    KNN_accuracy = accuracy_score(y, KNN_predictions)
    error = 1 - KNN_accuracy
    return error

KNN_error2 = KNN_error(X2,y2,KNN)

KNN_error1 = KNN_error(X1,y1,KNN)
KNN_error3 = KNN_error(X3,y3,KNN)
KNN_error4 = KNN_error(X4,y4,KNN)
KNN_error5 = KNN_error(X5,y5,KNN)
KNN_error_test = KNN_error(X_test,y_test,KNN)

KNN_data = np.vstack((KNN_error1,KNN_error2,KNN_error3,KNN_error4,KNN_error5,KNN_error_test))

#Part f
LogReg = LogisticRegression(multi_class = 'ovr')
LogReg.fit(X3,y3)

def LogReg_error(X,y,LogReg):
    LogReg_predictions = LogReg.predict(X)
    LogReg_accuracy = accuracy_score(y, LogReg_predictions)
    error = 1 - LogReg_accuracy
    return error

LogReg_error3 = LogReg_error(X3,y3,LogReg)

LogReg_error1 = LogReg_error(X1,y1,LogReg)
LogReg_error2 = LogReg_error(X2,y2,LogReg)
LogReg_error4 = LogReg_error(X4,y4,LogReg)
LogReg_error5 = LogReg_error(X5,y5,LogReg)
LogReg_error_test = LogReg_error(X_test,y_test,LogReg)

LogReg_data = np.vstack((LogReg_error1,LogReg_error2,LogReg_error3,LogReg_error4,LogReg_error5,LogReg_error_test))

#Part g 
DecTree = DecisionTreeClassifier()
DecTree.fit(X4,y4)

def DecTree_error(X,y,DecTree):
    DecTree_predictions = DecTree.predict(X)
    DecTree_accuracy = accuracy_score(y, DecTree_predictions)
    error = 1 - DecTree_accuracy
    return error

DecTree_error4 = DecTree_error(X4,y4,DecTree)

DecTree_error1 = DecTree_error(X1,y1,DecTree)
DecTree_error2 = DecTree_error(X2,y2,DecTree)
DecTree_error3 = DecTree_error(X3,y3,DecTree)
DecTree_error5 = DecTree_error(X5,y5,DecTree)
DecTree_error_test = DecTree_error(X_test,y_test,DecTree)

DecTree_data = np.vstack((DecTree_error1,DecTree_error2,DecTree_error3,DecTree_error4,DecTree_error5,DecTree_error_test))

#part h
rand_Forest = RandomForestClassifier(n_estimators = 85)
rand_Forest.fit(X4,y4)

def RandForest_Error(X,y,rand_Forest):
    rand_Forest_predictions = rand_Forest.predict(X)
    rand_Forest_accuracy = accuracy_score(y, rand_Forest_predictions)
    error = 1 - rand_Forest_accuracy
    return error

RandForest_Error5 = RandForest_Error(X5,y5,rand_Forest)

RandForest_Error1 = RandForest_Error(X1,y1,rand_Forest)
RandForest_Error2 = RandForest_Error(X2,y2,rand_Forest)
RandForest_Error3 = RandForest_Error(X3,y3,rand_Forest)
RandForest_Error4 = RandForest_Error(X4,y4,rand_Forest)
RandForest_Error_test = RandForest_Error(X_test,y_test,rand_Forest)

RandForest_data = np.vstack((RandForest_Error1,RandForest_Error2,RandForest_Error3,RandForest_Error4,RandForest_Error5,RandForest_Error_test))
#Part i
svm_predictions = svm.predict(X_test)
KNN_predictions = KNN.predict(X_test)
LogReg_predictions = LogReg.predict(X_test)
DecTree_predictions = DecTree.predict(X_test)
rand_Forest_predictions = rand_Forest.predict(X_test)

all_class_predicition = np.column_stack([svm_predictions,KNN_predictions,LogReg_predictions,DecTree_predictions,rand_Forest_predictions])
voted_prediction = stats.mode(all_class_predicition, axis = 1)[0].ravel()

voted_accraucy = accuracy_score(y_test,voted_prediction)
voted_error = 1 - voted_accraucy
print(voted_error)

header = ['SVM', 'KNN', 'LogReg', 'DecTree', 'RandForest', 'Voted Prediction']
row_names = [['data1'],['data2'],['data3'],['data4'],['data5'],['Test data']]
total_data = np.column_stack([row_names,svm_data,KNN_data,LogReg_data,DecTree_data,RandForest_data])

table = tabulate(total_data, headers=header,tablefmt="grid")