import pandas as pd
import numpy as np
import random
from model import model

import csv

def shuffle(df):
    index = list(df.index)
    random.shuffle(index)
    df = df.iloc[index]
    df.reset_index()
    return df

data = pd.read_csv('C:\\Users\\jaide\\OneDrive\\Documents\\VSCODE\\NeuralNets\\IRIS.csv')

species = np.unique(data['species'])
modify = {'Setosa':0,'Versicolor':1,'Virginica':2}
data.species = [modify[x] for x in data.species]

data =  shuffle(data)

train = data[:100]
test = data[100:]

x_train,y_train,x_test,y_test = train[['sl','sw','pl','pw']].values,train['species'].values,test[['sl','sw','pl','pw']].values,test['species'].values

change = [[1,0,0],[0,1,0],[0,0,1]]

y_train = np.array([change[int(x)] for x in y_train])
y_test = np.array([change[int(x)] for x in y_test])


model(x_train,y_train,x_test,y_test)

