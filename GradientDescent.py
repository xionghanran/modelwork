import numpy as np
import matplotlib.pyplot as plt

train_data = np.random.rand(1000, 2)
train_label = np.random.randint(0, 2, 1000)
wight = np.random.rand(3)
initial_weight = np.array([0.1, 0.2, 0.3])
#符号函数
def sign(x):
    if  x>0:
        return 1
    elif x<=0:
        return -1
    
#增广数据
def data_add_one(data):
    data = np.hstack((np.ones((len(data), 1)), data))
    return data

#损失函数
'''
def loss_function(data,label,weight):
    loss=0
    for i in range(len(data)):
        loss+=(label[i]-sign(np.dot(weight,data[i])))**2
    print(loss/len(data))
    return loss/len(data)
'''
def loss_function(data, label, weight):
    loss = 0
    for i in range(len(data)):
        loss += (label[i] - np.dot(weight, data[i])) ** 2
    #print(loss/len(data))
    return loss/len(data)
#梯度
def gradient(data,label,weight):
    grad=0
    for i in range(len(data)):
        grad+=(label[i]-sign(np.dot(weight,data[i])))*data[i]
    return grad/len(data)


#梯度下降
def gradient_descent(data, label, weight, initial_lr, decay_rate, stages):
    loss_list = []
    lr = initial_lr
    for i in range(stages):
        loss = loss_function(data, label, weight)
        loss_list.append(loss)
        if len(loss_list) > 2 and abs(loss_list[-1] - loss_list[-2]) < 0.000001:
            break
        grad = gradient(data, label, weight)
        weight -= lr * grad  # 更新权重
        lr *= decay_rate  # 学习率衰减
    return weight, loss_list


#可视化损失函数 
def visualize_loss(loss):
    plt.figure()
    x = np.arange(0, len(loss))
    plt.plot(x, loss)
    plt.xlabel('stage')
    plt.ylabel('loss_value')
    plt.title('Training Data')
    plt.show()


train_data = data_add_one(train_data)
weight,loss_list=gradient_descent(train_data,train_label,initial_weight,0.001,0.999,1000)
visualize_loss(loss_list)

