import numpy as np
import math

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def relu_dev(x):
    x[x<=0]=0
    x[x>0]=1
    return x

np.random.seed(42)

def prepare_weights(x,h,k,y):
    W1 = np.random.randn(x,h)-1
    W2 = np.random.randn(h,k)-1
    W3 = np.random.randn(k,y)-1

    return W1,W2,W3

def model(X,y,j,k):
    X = np.array(X)
    y = np.array(y)
    
    W1,W2,W3 = prepare_weights(13,12,12,1)

    print(X.shape)

    epochs = 1000
    learning_rate=0.05
    for x in range(epochs):
        A = np.dot(X,W1)
        B = relu(A)
        C = np.dot(B,W2)
        D = relu(C)
        E = np.dot(D,W3)
        F = E


        error = ((y-F)**2).mean()
        print(error)
        # print(F.shape)
        # print(y.shape)
        # print((y-F).shape)
        delF = 2(y-F)
        delD = delF.dot(W3.T)*relu_dev(D)
        delB = delD.dot(W2.T)*relu_dev(B)


        W3 = W3+(D.T.dot(delF)*learning_rate)
        W2 = W2+(B.T.dot(delD)*learning_rate)
        W1 = W1+(X.T.dot(delB)*learning_rate)

    print("error is  : "+str(error))


    A = np.dot(j,W1)
    B = relu(A)
    C = np.dot(B,W2)
    D = relu(C)
    E = np.dot(D,W3)
    F = E

    np.round(F,3)

    print(F)
    


 
    
        

    