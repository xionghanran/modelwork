import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([[1, 3, 3], [1, 4, 3], [1, 1, 3], [1, 3, 9]])
train_label = np.array([1, 1, -1, 1])
test_data = np.array([1, 1, 2])
weight = np.array([0, 1, 1])

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

lr = 1
stages = 100
weight = LogisticRegression(train_data, train_label, weight, lr, stages)
print("weight:", weight)