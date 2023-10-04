import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

# 加载MNIST数据集
mnist = fetch_openml(name='mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target
X = X / 255.0  # 像素值归一化到[0, 1]范围

# 将标签进行one-hot编码
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# 划分训练集和测试集
X, y = shuffle(X, y, random_state=42)
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

# Softmax算法实现多类别分类
def softmax(X, W, b):
    scores = np.dot(X, W) + b
    exp_scores = np.exp(scores)
    softmax_output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return softmax_output

def train_softmax(X_train, y_train, X_test, y_test, learning_rate=0.01, batch_size=256, epochs=10):
    num_classes = y_train.shape[1]
    num_features = X_train.shape[1]
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    # 初始化权重和偏置
    np.random.seed(42)
    W = np.random.normal(0, 0.01, (num_features, num_classes))
    b = np.zeros((num_classes,))

    # 保存训练过程的损失函数、训练集分类精度和测试集分类精度
    loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(epochs):
        # 在每个epoch之前将训练集洗牌
        X_train, y_train = shuffle(X_train, y_train)

        for batch in range(num_batches):
            # 获取当前batch的训练集和标签
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # 计算当前batch的预测值
            softmax_output = softmax(X_batch, W, b)

            # 计算当前batch的损失函数值
            loss = -np.mean(np.sum(y_batch * np.log(softmax_output), axis=1))
            loss_history.append(loss)

            # 计算当前batch的梯度
            gradient = softmax_output - y_batch
            dW = np.dot(X_batch.T, gradient)
            db = np.sum(gradient, axis=0)

            # 更新权重和偏置
            W -= learning_rate * dW
            b -= learning_rate * db

        # 在当前epoch结束后计算训练集和测试集分类精度
        train_predictions = np.argmax(softmax(X_train, W, b), axis=1)
        train_accuracy = np.mean(train_predictions == np.argmax(y_train, axis=1))
        train_acc_history.append(train_accuracy)

        test_predictions = np.argmax(softmax(X_test, W, b), axis=1)
        test_accuracy = np.mean(test_predictions == np.argmax(y_test, axis=1))
        test_acc_history.append(test_accuracy)

        # 打印当前epoch的损失函数和分类精度
        print("Epoch", epoch + 1, "- Loss:", loss, "- Train Accuracy:", train_accuracy, "- Test Accuracy:", test_accuracy)

    return W, b, loss_history, train_acc_history, test_acc_history

# 使用Softmax算法进行多类别分类训练
W, b, loss_history, train_acc_history, test_acc_history = train_softmax(X_train, y_train, X_test, y_test)

# 绘制损失函数随epoch变化的曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.show()



# 绘制训练集和测试集分类精度随epoch变化的曲线
plt.figure(figsize=(10, 5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch') 
plt.legend()
plt.show()

# 随机抽取10个样本进行观察
num_samples = 10
random_indices = np.random.choice(len(X_test), num_samples, replace=False)
samples = X_test[random_indices]
sample_labels = np.argmax(y_test[random_indices], axis=1)

# 使用训练得到的权重和偏置项对样本进行分类预测
sample_predictions = np.argmax(softmax(samples, W, b), axis=1)

# 打印样本的真实标签和预测标签
for i in range(num_samples):
    print("Sample", i + 1, "- True Label:", sample_labels[i], "- Predicted Label:", sample_predictions[i])