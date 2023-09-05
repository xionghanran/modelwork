import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([[0.2,0.7],
                      [0.3,0.3],
                      [0.4,0.5],
                      [0.6,0.5],
                      [0.1,0.4],
                      [0.4,0.6],
                      [0.6,0.2],
                      [0.7,0.4],
                      [0.8,0.6],
                      [0.7,0.5]])

train_label=np.array([1,1,1,1,1,-1,-1,-1,-1,-1])

wight=np.array([0,1,1])

def sign(x):
    if  x>=0:
        return 1
    elif x<0:
        return -1
    
def data_add_one(data):
    data = np.hstack((np.ones((len(data), 1)), data))
    return data

#可视化
def visualize(train_data,train_label,wight):
    plt.figure()
    plt.scatter(train_data[train_label==1,0],train_data[train_label==1,1],c='r')
    plt.scatter(train_data[train_label==-1,0],train_data[train_label==-1,1],c='b')
    x=np.linspace(0,1,100)
    y=(-wight[1]*x-wight[0])/wight[2]
    plt.plot(x,y)
    plt.show()


#去除增广数据
def data_remove_one(data):
    data = data[:, 1:]
    return data


def train_with_lra(train_data,train_label,wight):
    
    #计算train_data的广义逆矩阵
    train_data=data_add_one(train_data)
    new_train_data=np.linalg.pinv(train_data)
    #计算权重向量w
    wight=np.dot(new_train_data,train_label)

    return wight

print("result w:",train_with_lra(train_data,train_label,wight))


visualize(train_data,train_label,train_with_lra(train_data,train_label,wight))

    