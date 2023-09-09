import numpy as np
import matplotlib.pyplot as plt
import datetime

weight=np.array([0.0,1.0,1.0])

#生成数据
def prepare_data():
    np.random.seed(10)
    x1 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], 200)
    label1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 200)
    label2 = np.ones(len(x2)) * -1
    return x1, label1, x2, label2

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

#去除增广数据
def data_remove_one(data):
    data = data[:, 1:]
    return data


#lra算法
def train_with_lra(train_data,train_label,wight):
    #计算train_data的广义逆矩阵
    train_data=data_add_one(train_data)
    new_train_data=np.linalg.pinv(train_data)
    #print("new_train_data:",new_train_data)
    #计算权重向量w
    wight=np.dot(new_train_data,train_label)
    return wight

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


#可视化
def visualize_classification(X, y, w):
    # 可视化训练数据和标签
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', marker='o', label='Class 1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', marker='x', label='Class -1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Training Data')
    plt.legend()
    # 画出分类面
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    grid_points_aug = np.hstack((np.ones((grid_points.shape[0], 1)), grid_points))
    predicted_labels = np.sign(np.dot(grid_points_aug, w))
    predicted_labels = predicted_labels.reshape(xx1.shape)
    plt.contourf(xx1, xx2, predicted_labels, alpha=0.2, cmap='coolwarm')
    plt.show()




#主函数
if __name__ == '__main__':
    
    x1, label1, x2, label2 = prepare_data()
    train_data = np.vstack((x1[:160], x2[:160]))
    train_label = np.hstack((label1[:160], label2[:160]))
    test_data = np.vstack((x1[160:], x2[160:]))
    test_label = np.hstack((label1[160:], label2[160:]))
    #输出广义逆的结果
    print("train_with_lra of weight:",train_with_lra(train_data,train_label,weight))
    visualize_classification(train_data,train_label,train_with_lra(train_data,train_label,weight))
    #输出梯度下降的结果
    print("train_with_gd:weight:",gradient_descent(data_add_one(train_data),train_label,weight,0.1,0.9,1000)[0])  
    visualize_classification(train_data,train_label,gradient_descent(data_add_one(train_data),train_label,weight,0.1,0.9,1000)[0])

