import numpy as np
from sklearn.svm import SVC

# 训练集数据
X_train = np.array([
    [31.2304, 121.4737],  # 上海
    [23.1200, 113.2500],  # 广州
    [36.0671, 120.3826],  # 青岛
    [39.0837, 117.2008],  # 天津
    [39.9042, 116.4074],  # 北京
    [35.682839, 139.759455],  # 东京
    [34.693738, 135.502165],  # 大阪
    [43.061771, 141.354376],  # 札幌
    [33.590355, 130.401716],  # 福冈
    [35.3191, 139.5467]  # 镰仓
])

y_train = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])  # 中国为正类，日本为负类

# 测试集数据
X_test = np.array([[25.7422, 123.5705]])  # 钓鱼岛


classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)


predicted_label = classifier.predict(X_test)

if predicted_label == 1:
    print("钓鱼岛属于中国")
else:
    print("钓鱼岛属于日本")


support_vectors = classifier.support_vectors_
print("支持向量的坐标：")
print(support_vectors)