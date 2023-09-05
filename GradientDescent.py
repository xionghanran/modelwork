import numpy as np
import matplotlib.pyplot as plt

train_data = np.random.rand(100, 2)
train_label = np.random.randint(0, 2, 100)
wight = np.random.rand(3)

#符号函数
def sign(x):
    if  x>=0:
        return 1
    elif x<0:
        return -1
    
#增广数据
def data_add_one(data):
    data = np.hstack((np.ones((len(data), 1)), data))
    return data

