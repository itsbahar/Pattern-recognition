from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import utilities as utl


if __name__ == '__main__':

    mat_contents = sio.loadmat('./datasets/Dataset1.mat')
    Inputs = mat_contents['X']
    Targets = mat_contents['y'] 

    # Train and test split
    x_trn, x_tst, y_trn, y_tst = train_test_split(
                Inputs, Targets, test_size=0.2, random_state=42)
    y_trn = np.array(y_trn == 0, dtype=float)
    y_tst = np.array(y_tst == 0, dtype=float)
    y_trn[y_trn == 0] = -1
    y_tst[y_tst == 0] = -1
    X_trn = np.array(x_trn)
    x_tst = np.array(x_tst)
    y_trn = np.array(y_trn.reshape(-1, 1))
    y_tst = np.array(y_tst.reshape(-1, 1))

    # #linear svm
    C=[1,100,1000]
    for c in C:
        w,b = utl.svm(X_trn,y_trn,c)
        prediction = utl.predict(x_tst,w,b)
        print("C:",c)
        print("w, b:", [w, b])
        print("Accuracy test  :", accuracy_score(prediction,y_tst))
        utl.plot_linear_svm(x_tst, y_tst, w, b)
        prediction = utl.predict(X_trn,w,b)
        print("Accuracy train :", accuracy_score(prediction,y_trn))
        utl.plot_linear_svm(x_trn, y_trn, w, b)

    #nonlinear
    mat_contents = sio.loadmat('./datasets/Dataset2.mat')
    Inputs = mat_contents['X']
    Targets = mat_contents['y']
    Inputs=utl.RBF(Inputs,0.1)  
    # Train and test split
    x_trn, x_tst, y_trn, y_tst = train_test_split(
                Inputs, Targets, test_size=0.2, random_state=42)
    y_trn = np.array(y_trn == 0, dtype=float)
    y_tst = np.array(y_tst == 0, dtype=float)
    y_trn[y_trn == 0] = -1
    y_tst[y_tst == 0] = -1
    X_trn = np.array(x_trn)
    x_tst = np.array(x_tst)
    y_trn = np.array(y_trn.reshape(-1, 1))
    y_tst = np.array(y_tst.reshape(-1, 1))
    
    w,b=utl.svm (x_trn,y_trn,0.03)
    prediction=utl.predict(x_tst,w,b)
    acc=accuracy_score(prediction,y_tst)
    print(acc)

    # Accuracy = []
    # for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
    #     tmp=[]
    #     for Sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
    #         trn_new=utl.RBF(x_trn,Sigma)
    #         w,b=utl.svm (trn_new,y_trn,C)
    #         predict(X,w,b)
    #         acc=accuracy_score(prediction,y_trn)
    #         print(acc)
    #         tmp.append(acc)
    #     np.array(tmp)    
    # Accuracies.append(Accuracy)

