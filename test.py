import numpy as np
p = np.array([[1,-1,3],[-1,5,-7]])

print(p)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def relu_dev(x):
    x[x<=0]=0
    x[x>0]=1
    return x
p = relu_dev(p)

print(p)
