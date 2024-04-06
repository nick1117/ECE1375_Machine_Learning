import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from nnCost import nnCost
import matplotlib.pyplot as plt
import time


def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lam, alpha, MaxEpochs):
    #start = time.time()
    print('start SG')
    Theta1 = np.random.rand(input_layer_size, hidden_layer_size) * (2*0.18) - 0.18
    Theta2 = np.random.rand(hidden_layer_size+1, num_labels) * (2*0.18) - 0.18
    
    costs = []

    K = np.max(y_train)
    y_matrix = np.eye(K)[y_train - 1]
    m = X_train.shape[0]
    for epoch in range(MaxEpochs):
        #startm = time.time()
        for i in range(m):
            #start1 = time.time()
            #xi = X_train[i,:].reshape(1,-1) #reshape to work
            #yi = y_matrix[i,:].reshape(1,-1) #reshape to work
            xi = X_train[i] #1024,
            xi.shape = (len(xi),1)
            yi = y_matrix[i] #3,
            yi.shape = (len(yi),1)
            #end1 = time.time()

            #start2 = time.time()
            # a1 = np.hstack([np.ones((1, 1)), xi])
            # z2 = a1 @ Theta1
            # a2 = np.hstack([np.ones((1, 1)), sigmoid(z2)])
            # z3 = a2 @ Theta2
            # a3 = sigmoid(z3)
            a1 = np.vstack([np.ones((1, 1)), xi])
            z2 = Theta1.T @ a1
            a2 = np.vstack([np.ones((1, 1)), sigmoid(z2)])
            z3 = Theta2.T @ a2
            a3 = sigmoid(z3)
            #end2 = time.time()

            #start3 = time.time()
            del3 = a3 - yi
            #del2 = del3 @ Theta2[1:,:].T * sigmoidGradient(z2)
            del2 = Theta2[1:,:] @ del3 * sigmoidGradient(z2)
            
            #end3 = time.time()

            #start4 = time.time()
            #d1 = a1.T @ del2
            d1 =  del2 @ a1.T
            #d2 = a2.T @ del3
            d2 = del3 @ a2.T
            #end4 = time.time()

            #start5 = time.time()
            Theta1_grad = d1 + lam * np.hstack([np.zeros((Theta1.shape[1], 1)), Theta1[1:,:].T])
            Theta2_grad = d2 + lam * np.hstack([np.zeros((Theta2.shape[1], 1)), Theta2[1:,:].T])
            end5 = time.time()

            #start6 = time.time()
            Theta1 -= alpha * Theta1_grad.T
            Theta2 -= alpha * Theta2_grad.T
            #end6 = time.time()
        # endm = time.time()
        # print(f'1: {end1-start1}')
        # print(f'2: {end2-start2}')
        # print(f'3: {end3-start3}')
        # print(f'4: {end4-start4}')
        # print(f'5: {end5-start5}')
        # print(f'6: {end6-start6}')
        # print(f'm loop: {endm-startm}')
        cost = nnCost(Theta1.T, Theta2.T, X_train, y_train, K, lam)
        costs.append(cost)

    #end = time.time()
    #print(f'sGD time: {end - start}')
    plt.plot(range(1, MaxEpochs + 1), costs, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost vs Epoch')
    plt.savefig('output\\ps7-4-e-1.png')
    plt.show()
    print('ran SG')
    return Theta1, Theta2


