import numpy as np
weight = np.array([0.01, 0.02, 0.62])
#广义逆算法
def generalized_inverse_least_squares(X, y):
    # 计算广义逆
    X_pseudo_inv = np.linalg.pinv(X)
    # 计算最佳解
    weight = np.dot(X_pseudo_inv, y)
    return weight

def gradient_descent(X, y, lr, epoch):
    m = len(y)  # 样本数量
    weight = np.zeros(X.shape[1])  # 初始化参数
    for i in range(epoch):
        y_pred = np.dot(X, weight)  # 计算预测值
        error = y_pred - y  # 计算误差
        gradient = (1/m) * np.dot(X.T, error)  # 计算梯度
        weight = weight - lr * gradient  # 更新参数
    return weight

#主函数
if __name__ == '__main__':
    X = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 5], [1, 4, 7]])
    y = np.array([1, 2, 3, 6])
    weight = generalized_inverse_least_squares(X, y)
    print("generalized_inverse_least_squares of weight:", weight)
    weight = gradient_descent(X, y, 0.01, 2000)
    print("gradient_descent of weight:", weight)