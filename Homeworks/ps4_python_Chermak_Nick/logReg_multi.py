import numpy as np
from sklearn.linear_model import LogisticRegression


def logReg_multi(X_train, y_train, X_test):
    unique_values = np.unique(y_train)
    y_c = np.ones(len(y_train),)
    y_predict = np.zeros((len(X_test),))
    for c in unique_values:
        for i in range(len(y_train)):
            if(c == y_train[i]):
                y_c[i] = 1
            else:
                y_c[i] = 0
        mdl_c = LogisticRegression(random_state=0).fit(X_train, y_c)
        proba_c = mdl_c.predict_proba(X_test)[:, 1]
        
        for j in range(len(X_test)):
            if(proba_c[j] > 0.5):
                y_predict[j] = c
    return y_predict


#def logReg_multi(X_train, y_train, X_test):
#    unique_values = np.unique(y_train)
#    y_predict = np.zeros(len(X_test))
#    
#    for c in unique_values:
#        # Create binary labels for current class
#        y_c = (y_train == c).astype(int)
#        
#        # Train logistic regression model for current class
#        mdl_c = LogisticRegression(random_state=0).fit(X_train, y_c)
#        
#        # Predict probabilities for positive class
#        proba_c = mdl_c.predict_proba(X_test)[:, 1]
#        
#        # Update predictions based on maximum probability
#        y_predict[proba_c.argmax()] = c
#    
#    return y_predict