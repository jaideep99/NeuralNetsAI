import numpy as np
import math

np.random.seed(42)

def sigmoid(x):
    return 1/(1+np.exp(-x))



def prepare_weights(x,h,k,y):
    W1 = np.random.randn(x,h)-1
    W2 = np.random.randn(h,k)-1
    W3 = np.random.randn(k,y)-1

    return W1,W2,W3
def model(X,y,j,k):
    X = np.array(X)
    y = np.array(y)
    
    W1,W2,W3 = prepare_weights(4,12,12,3)

    print(X.shape)
    epochs = 10000
    learning_rate=0.05
    for x in range(epochs):
        A = np.dot(X,W1)
        B = sigmoid(A)
        C = np.dot(B,W2)
        D = sigmoid(C)
        E = np.dot(D,W3)
        F = sigmoid(E)

        error = ((y-F)**2).mean()

        delF = (y-F)*(F*(1-F))
        delD = delF.dot(W3.T)*(D*(1-D))
        delB = delD.dot(W2.T)*(B*(1-B))


        W3 = W3+(D.T.dot(delF)*learning_rate)
        W2 = W2+(B.T.dot(delD)*learning_rate)
        W1 = W1+(X.T.dot(delB)*learning_rate)

    print("error is  : "+str(error))


    A = np.dot(j,W1)
    B = sigmoid(A)
    C = np.dot(B,W2)
    D = sigmoid(C)
    E = np.dot(D,W3)
    F = sigmoid(E)

    np.round(F,3)    


    pred = np.argmax(F,axis=1)
    true = np.argmax(k,axis=1)

    print(pred)
    print(true)
    
    accuracy=0
    for i in range(50):
        if pred[i]==true[i]:
            accuracy+=1

    accuracy = (float(accuracy)/50)*100

    print(accuracy)

    
        

    