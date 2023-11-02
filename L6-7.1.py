import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#svm_primal模型
def svm_primal(X, y, C):
    n_samples, n_features = X.shape

    # 定义目标函数
    def objective(w):
        return 0.5 * np.dot(w, w) + C * np.sum(np.maximum(0, 1 - y * np.dot(X, w)))

    # 定义约束条件
    def constraint(w):
        return y * np.dot(X, w) - 1

    # 初始化权重向量
    w0 = np.zeros(n_features)

    # 定义约束条件的字典形式
    constraints = {'type': 'ineq', 'fun': constraint}

    # 使用二次规划函数求解
    res = minimize(objective, w0, constraints=constraints)

    # 提取最优权重向量
    w = res.x

    return w

#对偶模型
def svm_dual(X, y, C):
    n_samples, n_features = X.shape

    # 计算 Gram 矩阵
    gram_matrix = np.dot(X, X.T)

    # 定义目标函数
    def objective(alpha):
        return 0.5 * np.sum(np.outer(alpha, alpha) * gram_matrix) - np.sum(alpha)

    # 定义约束条件
    def constraint(alpha):
        return np.dot(alpha, y)

    # 定义不等式约束条件
    def inequality_constraint(alpha):
        return alpha - C

    # 初始化拉格朗日乘子向量
    alpha0 = np.zeros(n_samples)

    # 定义优化问题
    bounds = [(0, C)] * n_samples  # 拉格朗日乘子的上下界
    constraints = [{'type': 'eq', 'fun': constraint}, {'type': 'ineq', 'fun': inequality_constraint}]
    options = {'maxiter': 100, 'disp': False}
    result = minimize(objective, alpha0, method='SLSQP', bounds=bounds, constraints=constraints, options=options)

    # 获取最优拉格朗日乘子向量
    alpha = result.x

    # 计算权重向量
    weights = np.dot(alpha * y, X)

    return weights

#核函数模型
def svm_kernel(X, y, C, kernel):
    n_samples, n_features = X.shape

    # 计算 Gram 矩阵
    gram_matrix = kernel(X, X)

    # 定义目标函数
    def objective(alpha):
        return 0.5 * np.sum(np.outer(alpha, alpha) * gram_matrix) - np.sum(alpha)

    # 定义约束条件
    def constraint(alpha):
        return np.dot(alpha, y)

    # 定义不等式约束条件
    def inequality_constraint(alpha):
        return alpha - C

    # 初始化拉格朗日乘子向量
    alpha0 = np.zeros(n_samples)

    # 定义优化问题
    bounds = [(0, C)] * n_samples  # 拉格朗日乘子的上下界
    constraints = [{'type': 'eq', 'fun': constraint}, {'type': 'ineq', 'fun': inequality_constraint}]
    options = {'maxiter': 100, 'disp': False}
    result = minimize(objective, alpha0, method='SLSQP', bounds=bounds, constraints=constraints, options=options)

    # 获取最优拉格朗日乘子向量
    alpha = result.x

    # 提取支持向量的索引
    support_vector_indices = np.where(alpha > 1e-5)[0]

    # 提取支持向量和对应的标签
    support_vectors = X[support_vector_indices]
    support_vector_labels = y[support_vector_indices]
    support_vector_alphas = alpha[support_vector_indices]

    # 计算权重向量
    weights = np.sum(support_vector_alphas * support_vector_labels[:, np.newaxis] * kernel(support_vectors, X), axis=0)

    return weights

# 示例核函数：高斯核函数
def gaussian_kernel(X1, X2, sigma=0.1):
    pairwise_dists = np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2)
    return np.exp(-pairwise_dists / (2 * sigma ** 2))


# 可视化数据和决策边界
def plot_decision_boundary(X, y, weights):
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

    # 绘制决策边界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))
    Z = np.sign(np.dot(np.c_[xx1.ravel(), xx2.ravel()], weights))
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, colors='k', levels=[-1, 0, 1], alpha=0.5)
    # 设置图形属性
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()


def prepare_data():
    np.random.seed(0)
    x1 = np.random.multivariate_normal([3, 0], [[1, 0], [0, 1]], 200)
    label1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal([0, 3], [[1, 0], [0, 1]], 200)
    label2 = np.ones(len(x2)) * -1
    return x1, label1, x2, label2

#统计正确率
def accuracy(X, y, weights):
    y_pred = np.sign(np.dot(X, weights))
    return np.sum(y_pred == y) / len(y)

if __name__ == '__main__':
    x1, label1, x2, label2 = prepare_data()
    train_data = np.vstack((x1[:160], x2[:160]))
    train_label = np.hstack((label1[:160], label2[:160]))
    test_data = np.vstack((x1[160:], x2[160:]))
    test_label = np.hstack((label1[160:], label2[160:]))
    C=1.0
    

    # 调用函数进行训练
    weights = svm_primal(train_data, train_label, C)
    print("svm_primal权重向量:", weights)
    print("svm_primal训练集正确率:", accuracy(train_data, train_label, weights))
    #可视化数据和决策边界
    plot_decision_boundary(train_data, train_label, weights)

    weights = svm_dual(train_data, train_label, C)
    print("svm_dual权重向量:", weights)
    print("svm_dual训练集正确率:", accuracy(train_data, train_label, weights))
    
    #可视化数据和决策边界
    plot_decision_boundary(train_data, train_label, weights)
    '''
    weights = svm_kernel(train_data, train_label, C, gaussian_kernel)
    print("svm_kernel权重向量:", weights)
    
    #可视化数据和决策边界
    plot_decision_boundary2(train_data, train_label, weights, gaussian_kernel)
    '''

   