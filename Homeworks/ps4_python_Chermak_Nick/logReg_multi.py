import numpy as np
from sklearn.linear_model import LogisticRegression


def logReg_multi(X_train, y_train, X_test):
    unique_values = np.unique(y_train)
    y_c = np.ones(len(y_train),)
    y_predict = []
    prob_total = []
    for c in unique_values:
        for i in range(len(y_train)):
            if(c == y_train[i]):
                y_c[i] = 1
            else:
                y_c[i] = 0
        mdl_c = LogisticRegression(random_state=0).fit(X_train, y_c)
        proba_c = mdl_c.predict_proba(X_test)[:, 1]
        prob_total.append(proba_c)
        #for j in range(len(X_test)):
        #    if(proba_c[j] > 0.5):
        #        y_predict[j] = c
    for row in np.transpose(prob_total):
        predictedClass = np.argmax(row) + 1
        y_predict.append(predictedClass)
    return y_predict

