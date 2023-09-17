import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# 原问题求解的支持向量机算法（Primal-SVM）
def primal_svm(X, y):
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    return clf

# 对偶的支持向量机算法（Dual-SVM）
def dual_svm(X, y):
    clf = svm.SVC(kernel='linear', C=1e10)  # 设置一个很大的C，使其接近对偶问题的解
    clf.fit(X, y)
    return clf

# 核函数的支持向量机算法（Kernel-SVM）
def kernel_svm(X, y):
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, y)
    return clf

# 示例用法
X = np.array([[0, 0], [1, 1]])
y = np.array([0, 1])

# 原问题求解的支持向量机算法
primal_model = primal_svm(X, y)
dual_model = dual_svm(X, y)
kernel_model = kernel_svm(X, y)
clf = dual_model
print("Primal-SVM:")
print("Support vectors:")
print(primal_model.support_vectors_)
print("Coefficients:")
print(primal_model.coef_)
print("Intercept:")
print(primal_model.intercept_)
'''
# 对偶的支持向量机算法
dual_model = dual_svm(X, y)
print("\nDual-SVM:")
print("Support vectors:")
print(dual_model.support_vectors_)
print("Dual coefficients:")
print(dual_model.dual_coef_)
print("Intercept:")
print(dual_model.intercept_)

# 核函数的支持向量机算法
kernel_model = kernel_svm(X, y)
print("\nKernel-SVM:")
print("Support vectors:")
print(kernel_model.support_vectors_)
print("Dual coefficients:")
print(kernel_model.dual_coef_)
print("Intercept:")
print(kernel_model.intercept_)
'''

# 绘制样本点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# 获取分类面的斜率和截距
w = clf.coef_[0]
b = clf.intercept_[0]

# 绘制分类面直线
x = np.linspace(-0.5, 1.5, 100)
y = -(w[0] * x + b) / w[1]
plt.plot(x, y, '-')

# 设置坐标轴标签
plt.xlabel('X1')
plt.ylabel('X2')

# 显示图形
plt.show()