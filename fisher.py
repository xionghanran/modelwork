import numpy as np
import matplotlib.pyplot as plt

train_data_one = np.array([[5,37],[7,30],[10,35],[11.5,40],[14,38],[12,31]])
train_label_one=np.array([1,1,1,1,1,1])
train_data_two = np.array([[35,21.5],[39,21.7],[34,16],[37,17]])
train_label_two=np.array([-1,-1,-1,-1])
weight=np.array([0,1,1])
test_data=np.array([[10,30],[20,20],[30,10],[40,0]])

#计算样本均值
def mean(data):
    return np.mean(data,axis=0)

#计算样本协方差矩阵
def cov(data):
    return np.cov(data.T)

#fisher算法
def fisher(train_data_one,train_label_one,train_data_two,train_label_two,weight):
    #计算类内总离差阵
    sw=cov(train_data_one)+cov(train_data_two)
    #计算类间总离差阵的逆
    sw_inv=np.linalg.inv(sw)
    #计算weight
    weight = np.dot(sw_inv,mean(train_data_one)-mean(train_data_two))
    print("weight:",weight)
    return weight

#可视化
def visualize_classification(train_data_one, train_data_two, weight):
    plt.figure()
    plt.scatter(train_data_one[:, 0], train_data_one[:, 1],c='b', marker='o', label='Class 1')
    plt.scatter(train_data_two[:, 0], train_data_two[:, 1],c='r', marker='x', label='Class -1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    #画出分类面
    x=np.linspace(-40,40,100)
    y=(-weight[0]*x)/weight[1]
    plt.plot(x,y,label='classification surface of fisher')
    plt.legend()
    plt.show()

#测试样本算法
def test(test_data,weight):
    #计算判别门限
    threshold=np.dot(weight,mean(train_data_one)+mean(train_data_two))/2
    print("threshold:",threshold)
    #计算测试样本的类别
    for i in range(len(test_data)):
        if np.dot(weight,test_data[i])>threshold:
            print("test_data",test_data[i],"is in class one")
        else:
            print("test_data",test_data[i],"is in class two")

#测试
test(test_data,fisher(train_data_one,train_label_one,train_data_two,train_label_two,weight))
#展示
visualize_classification(train_data_one, train_data_two, 
                         fisher(train_data_one,train_label_one,train_data_two,train_label_two,weight))
