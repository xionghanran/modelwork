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

#去除增广数据
def data_remove_one(data):
    data = data[:, 1:]
    return data

#lra算法
def train_with_lra(train_data,train_label,wight):
    
    #计算train_data的广义逆矩阵
    train_data=data_add_one(train_data)
    new_train_data=np.linalg.pinv(train_data)
    #计算权重向量w
    wight=np.dot(new_train_data,train_label)

    return wight

print("result w:",train_with_lra(train_data,train_label,wight))

visualize_classification(train_data,train_label,train_with_lra(train_data,train_label,wight))

    