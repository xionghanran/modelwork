import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([[1,0.2,0.7]
                       ,[1,0.3,0.3]
                       ,[1,0.4,0.5]
                       ,[1,0.6,0.5]
                       ,[1,0.1,0.4]
                       ,[1,0.4,0.6]
                       ,[1,0.6,0.2]
                       ,[1,0.7,0.4]
                       ,[1,0.8,0.6]
                       ,[1,0.7,0.5]])

train_label = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
w0 = [1,1,1] # initial w
pocket_w = [1,1,7]

def sign(x):
    if  x>0:
        return 1
    elif x<=0:
        return -1


def predict(train_data,train_label,w0):
    n=train_data.shape[0]
    error=0
    for i in range(n):
        if(train_label[i]*np.dot(w0,train_data[i])<0):
            error+=1
    #print("error:",error)
    return error

def train_with_pocket(train_data,train_label,w0,pocket_w):
    for j in range(20):
        for i in range(len(train_data)):
            if(1):
                
                if(np.sign(np.dot(w0,train_data[i]))!=train_label[i]):
                    w_=w0+train_label[i]*train_data[i]
                    if(predict(train_data,train_label,w0)>predict(train_data,train_label,w_)):
                        pocket_w=w_
                    w0=w_
            
        print("result w:",w0) 

    return pocket_w

pocket_w = train_with_pocket(train_data,train_label,w0,pocket_w) 
print("error:",predict(train_data,train_label,pocket_w))   
print("result pocket_w:",pocket_w) 

