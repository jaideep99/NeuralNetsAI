import pandas as pd
import numpy as np
from network import model

data = pd.read_csv('C:\\Users\\jaide\\OneDrive\\Documents\\VSCODE\\NeuralNets\\Housing.csv')

columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTR','B','LSTAT','MEDB']


data = data.values

features = []
labels = []

for x in data:
    lt = []
    for y in x[0].split(' '):
        if y!='':
            lt.append(float(y))

    features.append(lt[:-1])
    labels.append([lt[13]])

features = np.array(features)
labels = np.array(labels)


x_train,y_train,x_test,y_test = features[:400],labels[:400],features[400:],labels[400:]

model(x_train,y_train,x_test,y_test)