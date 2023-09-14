import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    for _ in range(num_iterations):
        linear_model = np.dot(X, weights)
        y_pred = sigmoid(linear_model)
        
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        
        weights -= learning_rate * dw
    
    return weights

def predict_logistic_regression(X, weights):
    linear_model = np.dot(X, weights)
    y_pred = sigmoid(linear_model)
    y_pred_class = np.where(y_pred > 0.5, 1, -1)
    return y_pred_class

train_data = np.array([[1, 3, 3], [1, 4, 3], [1, 1, 3], [1, 3, 9]])
train_label = np.array([1, 1, -1, 1])
test_data = np.array([1, 1, 2])
w0 = np.array([0, 1, 1])

weights = train_logistic_regression(train_data, train_label, learning_rate=0.1, num_iterations=1000)
test_result = predict_logistic_regression(test_data, weights)
print("Test result:", test_result)