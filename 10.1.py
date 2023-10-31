import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 加载IRIS数据集
iris = load_iris()
X, y = iris.data, iris.target

# 将标签进行one-hot编码
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)

# 定义神经网络类
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.zeros((1, self.output_dim))

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

    def backward(self, X, y, learning_rate):
        # 反向传播
        m = X.shape[0]

        dZ2 = self.a2 - y
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0)

        # 更新权重和偏置
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.a2, axis=1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == np.argmax(y, axis=1))

# 设置超参数
input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = y_train.shape[1]
learning_rate = 0.01
num_epochs = 1000

# 创建神经网络对象
network = NeuralNetwork(input_dim, hidden_dim, output_dim)

# 训练神经网络
for epoch in range(num_epochs):
    network.forward(X_train)
    network.backward(X_train, y_train, learning_rate)

    # 每100个epoch打印一次训练集上的准确率
    if epoch % 100 == 0:
        train_acc = network.accuracy(X_train, y_train)
        print(f"Epoch {epoch}: Train Accuracy = {train_acc}")

# 在测试集上评估模型
test_acc = network.accuracy(X_test, y_test)
print(f"Test Accuracy: {test_acc}")