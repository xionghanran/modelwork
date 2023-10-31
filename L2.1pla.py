#pla算法的python实现
import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([[1,3,3],[1,4,3],[1,1,3],[1,3,9]])
train_label=np.array([1,1,-1,1])
test_data   =np.array([1,1,2])
w0=np.array([0,1,1])

def sign(x):
    if  x>0:
        return 1
    elif x<0:
        return -1
    
def train(train_data,train_label,w0):
    w=w0
    while True:
        count=0
        for i in range(len(train_data)):
            if sign(np.dot(w,train_data[i]))!=train_label[i]:
                w=w+train_label[i]*train_data[i]
                count+=1
        if count==0:
            break
    return w

w = train(train_data,train_label,w0)
print("result w:",w)
print("test result:",sign(np.dot(w,test_data)))
    
