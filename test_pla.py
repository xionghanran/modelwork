import numpy as np
import matplotlib.pyplot as plt
import datetime

inital_pla_weight = np.array([10.5, -0.32, 0.10])  # 初始化pla算法的权重
inital_pocket_weight = np.array([0.05, 0.106, 10.07])  # 初始化pocket算法的权重

#符号函数
def sign(x):
    if  x>0:
        return 1
    elif x<=0:
        return -1

#计算错误数
def predict(train_data,train_label,w0):
    n=train_data.shape[0]
    error=0
    for i in range(n):
        if(train_label[i]*np.dot(w0,train_data[i])<0):
            error+=1
    #print("error:",error)
    return error

#pocket算法
def train_with_pocket(train_data,train_label,w0,pocket_w):
    start = datetime.datetime.now()
    for j in range(20):
        for i in range(len(train_data)):
            if(1):
                if(np.sign(np.dot(w0,train_data[i]))!=train_label[i]):
                    w=w0+train_label[i]*train_data[i]
                    if(predict(train_data,train_label,w0)>predict(train_data,train_label,w)):
                        pocket_w=w
                    w0=w
        #print("result w:",w0)
    end = datetime.datetime.now()
    print('totally time of pocket is ', end - start)
    return pocket_w


#pla算法
def train_with_pla(train_data,train_label,w0):
    start = datetime.datetime.now()
    w=w0
    for j in range(20):
        
        for i in range(len(train_data)):
            if sign(np.dot(w,train_data[i]))!=train_label[i]:
                w=w+train_label[i]*train_data[i]
    end = datetime.datetime.now()
    print('totally time of pla is ', end - start)
    return w


#生成数据
def prepare_data():
    np.random.seed(10)
    x1 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], 200)
    label1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 200)
    label2 = np.ones(len(x2)) * -1
    return x1, label1, x2, label2

#增广数据
def data_add_one(data):
    data = np.hstack((np.ones((len(data), 1)), data))
    return data

#去除增广数据
def data_remove_one(data):
    data = data[:, 1:]
    return data



#可视化数据和分类面
def visualize_data_and_classfication_surface(X, y, weight_pla,weight_pocket):
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', marker='o', label='Class 1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', marker='x', label='Class -1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Testing Data')
    plt.legend()
    x=np.linspace(-10,10,100)
    y1=(-weight_pla[0]-weight_pla[1]*x)/weight_pla[2]
    y2=(-weight_pocket[0]-weight_pocket[1]*x)/weight_pocket[2]
    plt.plot(x,y1,label='classification surface of pla')
    plt.plot(x,y2,label='classification surface of pocket')
    plt.legend()
    plt.show()

#统计分类正确率
def calculate_accuracy(data,label,weight):
    error=0
    for i in range(len(data)):
        if sign(np.dot(weight,data[i]))!=label[i]:
            error+=1
    return 1-error/len(data)

    

#主函数
if __name__ == '__main__':
    x1, label1, x2, label2 = prepare_data()
    x1 = data_add_one(x1)
    x2 = data_add_one(x2)
    train_data = np.vstack((x1[:160], x2[:160]))
    train_label = np.hstack((label1[:160], label2[:160]))
    test_data = np.vstack((x1[160:], x2[160:]))
    test_label = np.hstack((label1[160:], label2[160:]))

    w_pocket=train_with_pocket(train_data,train_label,inital_pocket_weight,inital_pocket_weight)
    w_pla=train_with_pla(train_data,train_label,inital_pla_weight)

    print("w_pla:",w_pla)
    print("w_pocket:",w_pocket)

    print("accuracy of train of pla:",calculate_accuracy(train_data,train_label,w_pla))
    print("accuracy of train of pocket:",calculate_accuracy(train_data,train_label,w_pocket))
    
    print("accuracy of test of pla:",calculate_accuracy(test_data,test_label,w_pla))
    print("accuracy of test of pocket:",calculate_accuracy(test_data,test_label,w_pocket))
    test_data = data_remove_one(test_data)

    visualize_data_and_classfication_surface(test_data, test_label,w_pla, w_pocket)
    
  