import numpy as np
import matplotlib.pyplot as plt

weight = np.array([0.1, 0.2, 0.62])

#生成数据
def prepare_data():
    np.random.seed(10)
    x1 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], 200)
    label1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 200)
    label2 = np.ones(len(x2)) * -1
    return x1, label1, x2, label2

#符号函数
def sign(x):
    if  x>0:
        return 1
    elif x<=0:
        return -1
    
#增广数据
def data_add_one(data):
    data = np.hstack((np.ones((len(data), 1)), data))
    return data

#去除增广数据
def data_remove_one(data):
    data = data[:, 1:]
    return data

#计算样本均值
def mean(data):
    return np.mean(data,axis=0)

#计算样本协方差矩阵
def cov(data):
    return np.cov(data.T)

#可视化样本和分类面
def visualize_classification(X, y, weight):
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', marker='o', label='Class 1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', marker='x', label='Class -1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    if(len(X)==320):
        plt.title('Training Data')
    else:
        plt.title('Testing Data')
    plt.legend()
    x=np.linspace(-10,10,100)
    y=(-weight[0]*x)/weight[1]
    plt.plot(x,y,label='classification surface of fisher')
    plt.legend()
    plt.show()

#统计分类正确率
def calculate_accuracy(data,label,weight):
    error=0
    for i in range(len(data)):
        if sign(np.dot(weight,data[i]))!=label[i]:
            error+=1
    return 1-error/len(data)

#fisher算法
def fisher(data,label,weight):
    #计算类内总离差阵
    sw=cov(data)
    #计算类间总离差阵的逆
    sw_inv=np.linalg.inv(sw)
    #计算weight
    weight = np.dot(sw_inv,mean(data[0:160])-mean(data[160:320]))
    #print("weight:",weight)
    return weight

def test(test_data,label,weight):
    #计算判别门限
    threshold=np.dot(weight,mean(test_data[0:40])+mean(test_data[40:80]))/2.0
    print("threshold:",threshold)
    #计算测试样本的类别
    for i in range(len(test_data)):
        if np.dot(weight,test_data[i])>threshold:
            print("test_data",test_data[i],"is in label 1")
        else:
            print("test_data",test_data[i],"is in label -1")

#主函数
if __name__ == '__main__':
    x1, label1, x2, label2 = prepare_data()
    train_data = np.vstack((x1[:160], x2[:160]))
    train_label = np.hstack((label1[:160], label2[:160]))
    test_data = np.vstack((x1[160:], x2[160:]))
    test_label = np.hstack((label1[160:], label2[160:]))
    print("weight:",fisher(train_data,train_label,weight))
    print("calculate_accuracy of fisher:",calculate_accuracy(train_data,train_label,fisher(train_data,train_label,weight)))
    test(test_data,test_label,fisher(train_data,train_label,weight))
    visualize_classification(train_data,train_label,fisher(train_data,train_label,weight))
