import numpy as np
from scipy.optimize import minimize

def svm_gaussian_kernel(X, y, C, gamma):
    n_samples, n_features = X.shape

    # 计算高斯核矩阵
    def gaussian_kernel_matrix(X):
        pairwise_sq_dists = np.square(np.linalg.norm(X[:, np.newaxis] - X, axis=2))
        K = np.exp(-gamma * pairwise_sq_dists)
        return K

    # 定义目标函数
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)

    # 定义约束条件
    def constraint(alpha):
        return np.dot(y, alpha)

    # 初始化拉格朗日乘子向量
    alpha0 = np.zeros(n_samples)

    # 计算高斯核矩阵
    K = gaussian_kernel_matrix(X)

    # 定义约束条件的字典形式
    constraints = {'type': 'eq', 'fun': constraint}

    # 使用二次规划函数求解
    res = minimize(objective, alpha0, constraints=constraints)

    # 提取最优拉格朗日乘子向量
    alpha = res.x

    # 计算权重向量
    w = np.dot(alpha * y, X)

    # 找到支持向量
    support_vectors = X[alpha > 0]

    return w, support_vectors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 生成示例数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练支持向量机模型
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# 绘制分类面和样本
plt.figure(figsize=(8, 6))

# 绘制样本点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

# 绘制分类面
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格来评估模型
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# 绘制分类面的等高线
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# 设置坐标轴标签
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 设置图例
plt.legend(['Class 0', 'Class 1'])

# 显示图形
plt.show()