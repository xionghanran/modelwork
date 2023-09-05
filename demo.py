import numpy as np
import matplotlib.pyplot as plt

# 定义训练样本集
X = np.array([[0.2, 0.7],
              [0.3, 0.3],
              [0.4, 0.5],
              [0.6, 0.5],
              [0.1, 0.4],
              [0.4, 0.6],
              [0.6, 0.2],
              [0.7, 0.4],
              [0.8, 0.6],
              [0.7, 0.5]])

y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

# 增广样本矩阵
X_aug = np.hstack((np.ones((X.shape[0], 1)), X))

# 计算广义逆
X_pinv = np.linalg.pinv(X_aug)

# 计算权重向量
w = np.dot(X_pinv, y)

# 定义分类函数
def classify(x):
    x_aug = np.hstack((1, x))
    result = np.sign(np.dot(w, x_aug))
    return result

# 打印权重向量
print("权重向量：", w)

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

visualize_classification(X, y, w)

# 测试分类函数
test_samples = np.array([[0.25, 0.6],
                         [0.35, 0.35],
                         [0.55, 0.45],
                         [0.65, 0.3]])

for sample in test_samples:
    label = classify(sample)
    print("样本", sample, "的分类结果：", label)

