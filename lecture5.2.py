import numpy as np
import matplotlib.pyplot as plt
inital_Logistic_weight = np.array([0, 1, 1])
#生成数据
def prepare_data():
    np.random.seed(10)
    x1 = np.random.multivariate_normal([2, 0], [[1, 0], [0, 1]], 200)
    label1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal([0, 2], [[1, 0], [0, 1]], 200)
    label2 = np.ones(len(x2)) * -1
    return x1, label1, x2, label2

#增广数据
def data_add_one(data):
    data = np.hstack((np.ones((len(data), 1)), data))
    return data

#去除增广数据
def data_remove_one(data):
    data = data[:, 1:]
    return data

#符号函数
def sign(x):
    if  x>0:
        return 1
    elif x<=0:
        return -1

# sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 梯度
def gradient(data, label, weight):
    grad = 0
    for i in range(len(data)):
        grad += sigmoid(-label[i] * np.dot(weight, data[i])) * (-label[i]) * data[i]
    return grad / len(data)


# logistic 回归梯度下降算法
def LogisticRegression(train_data, train_label, weight, lr, stages):
    w = weight
    for i in range(stages):
        for j in range(len(train_data)):
            w = w - lr * gradient(train_data, train_label, w)
    return w

#可视化数据和分类面
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

#统计分类正确率
def calculate_accuracy(data,label,weight):
    error=0
    for i in range(len(data)):
        if sign(np.dot(weight,data[i]))!=label[i]:
            error+=1
    return 1-error/len(data)

def predict(train_data,train_label,w0):
    n=train_data.shape[0]
    error=0
    for i in range(n):
        if(train_label[i]*np.dot(w0,train_data[i])<0):
            error+=1
    #print("error:",error)
    return error

#主函数
if __name__ == '__main__':

    x1, label1, x2, label2 = prepare_data()
    x1 = data_add_one(x1)
    x2 = data_add_one(x2)
    train_data = np.vstack((x1[:160], x2[:160]))
    train_label = np.hstack((label1[:160], label2[:160]))
    test_data = np.vstack((x1[160:], x2[160:]))
    test_label = np.hstack((label1[160:], label2[160:]))
    #print("hello world")
    lr = 0.01
    stages = 20
    w_logistic = LogisticRegression(train_data, train_label, inital_Logistic_weight, lr, stages)
    print("w_logistic:", w_logistic)
    print("accuracy of train of logistic:", calculate_accuracy(train_data, train_label, w_logistic))
    visualize_classification(data_remove_one(train_data), train_label, w_logistic)


