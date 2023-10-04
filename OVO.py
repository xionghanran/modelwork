import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载IRIS数据集
iris = load_iris()
X = iris.data
y = iris.target

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# (a) 感知器算法 - OvO多类分类器
class OvoPerceptron:
    def __init__(self, classes):
        self.classes = classes
        self.classifiers = {}

    def train(self, X_train, y_train):
        for i in range(len(self.classes)):
            for j in range(i+1, len(self.classes)):
                class_1 = self.classes[i]
                class_2 = self.classes[j]
                X_binary = X_train[(y_train == class_1) | (y_train == class_2)]
                y_binary = y_train[(y_train == class_1) | (y_train == class_2)]
                y_binary = np.where(y_binary == class_1, -1, 1)
                clf = Perceptron()
                clf.fit(X_binary, y_binary)
                self.classifiers[(class_1, class_2)] = clf

    def predict(self, X_test):
        predictions = np.zeros((X_test.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            for j in range(i+1, len(self.classes)):
                class_1 = self.classes[i]
                class_2 = self.classes[j]
                clf = self.classifiers[(class_1, class_2)]
                binary_pred = clf.predict(X_test)
                binary_pred = np.where(binary_pred == -1, class_1, class_2)
                predictions[:, class_1] += np.where(binary_pred == class_1, 1, 0)
                predictions[:, class_2] += np.where(binary_pred == class_2, 1, 0)

        final_pred = np.argmax(predictions, axis=1)
        return final_pred

# 创建OvO多类分类器
ovo_classifier = OvoPerceptron(classes=[0, 1, 2])

# 训练OvO多类分类器
ovo_classifier.train(X_train, y_train)

# 在测试集上进行预测
ovo_predictions = ovo_classifier.predict(X_test)

# 计算准确率
ovo_accuracy = accuracy_score(y_test, ovo_predictions)
print("OvO多类分类器准确率：", ovo_accuracy)


# (b) Softmax算法
class SoftmaxClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.weights = None

    def train(self, X_train, y_train, num_epochs=100, learning_rate=0.1):
        num_features = X_train.shape[1]
        self.weights = np.zeros((num_features, self.num_classes))

        for epoch in range(num_epochs):
            scores = np.dot(X_train, self.weights)
            exp_scores = np.exp(scores)
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            delta = np.zeros(probabilities.shape)
            delta[np.arange(X_train.shape[0]), y_train] = 1
            gradient = np.dot(X_train.T, probabilities - delta)
            self.weights -= learning_rate * gradient

    def predict(self, X_test):
        scores = np.dot(X_test, self.weights)
        softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        predictions = np.argmax(softmax_output, axis=1)
        return predictions

# 创建Softmax分类器
softmax_classifier = SoftmaxClassifier(num_classes=3)

# 训练Softmax分类器
softmax_classifier.train(X_train, y_train)

# 在测试集上进行预测
softmax_predictions = softmax_classifier.predict(X_test)

# 计算准确率
softmax_accuracy = accuracy_score(y_test, softmax_predictions)
print("Softmax分类器准确率：", softmax_accuracy)