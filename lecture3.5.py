import numpy as np
import matplotlib.pyplot as plt

# f(x)=x*cos⁡(0.25π*x)
#原函数
def f(x):
    return x * np.cos(0.25 * np.pi * x)

# 梯度下降函数
def gradient_descent(x_init, learning_rate, num_iterations):

    x_values = [x_init]
    f_values = [f(x_init)]
    
    for i in range(num_iterations):
        #取最后一个x值
        x = x_values[-1]
        #计算梯度
        gradient = np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)
        #计算新的x值
        x_new = x - learning_rate * gradient
        #print('x_new is:', x_new)
        x_values.append(x_new)
        f_values.append(f(x_new))
    return x_values, f_values

#随机梯度下降函数   
def stochastic_gradient_descent(x_init, learning_rate, num_iterations):
    x_values = [x_init]
    f_values = [f(x_init)]
    n_samples = len(x_init)
    
    for i in range(num_iterations):
        # 随机选择一个样本
        random_index = np.random.randint(0, n_samples)
        x = x_values[-1]
        
        # 计算梯度
        gradient = np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)
        
        # 更新参数
        x_new = x - learning_rate * gradient
        x_new[random_index] = x_new[random_index] - learning_rate * gradient[random_index]
        
        x_values.append(x_new)
        f_values.append(f(x_new))
    
    return x_values, f_values

#adagrad函数
def adagrad(x_init, learning_rate, num_iterations):
    x_values = [x_init]
    f_values = [f(x_init)]
    n_samples = len(x_init)
    epsilon = 1e-8  # 避免除以0
    
    # 初始化累积梯度平方和
    gradient_sum_squared = np.zeros(n_samples)
    
    for i in range(num_iterations):
        # 随机选择一个样本
        #random_index = np.random.randint(0, n_samples)
        x = x_values[-1]
        
        # 计算梯度
        gradient = np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)
        
        # 更新累积梯度平方和
        gradient_sum_squared += gradient**2
        
        # 计算学习率
        learning_rate_adjusted = learning_rate / (np.sqrt(gradient_sum_squared) + epsilon)
        
        # 更新参数
        x_new = x - learning_rate_adjusted * gradient
        
        x_values.append(x_new)
        f_values.append(f(x_new))
    
    return x_values, f_values

#RMSProp函数
def RMSProp(x_init, learning_rate, num_iterations):
        
            x_values = [x_init]
            f_values = [f(x_init)]
            n_samples = len(x_init)
            epsilon = 1e-8  # 避免除以0
            # 初始化累积梯度
            gradient_sum = np.zeros(n_samples)
            for i in range(num_iterations):
                # 随机选择一个样本
                random_index = np.random.randint(0, n_samples)
                x = x_values[-1][random_index]
                # 计算梯度
                gradient = np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)
                # 更新累积梯度
                gradient_sum = 0.9 * gradient_sum ** 2 + 0.1 * gradient ** 2
                # 更新参数
                x_new = x_values[-1].copy()
                #x_new[random_index] = x - learning_rate * gradient / (np.sqrt(gradient_sum) + +epsilon)
                x_new = x - learning_rate * gradient / (np.sqrt(gradient_sum) + +epsilon)
                
                x_values.append(x_new)
                f_values.append(f(x_new))
            return x_values, f_values

#Momentum函数
def Momentum(x_init, learning_rate, num_iterations):
    x_values = [x_init]
    f_values = [f(x_init)]
    m0=0
    λ=0.9
    
    for i in range(num_iterations):
        #取最后一个x值
        x = x_values[-1]
        #计算梯度
        gradient = np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)
        #计算新的M值
        M_new = λ * m0 - learning_rate * gradient
        #计算新的x值
        x_new = x + M_new
        #print('x_new is:', x_new)
        x_values.append(x_new)
        f_values.append(f(x_new))
    return x_values, f_values

#Adam函数 
def Adam(x_init, learning_rate, num_iterations):
    x_values = [x_init]
    f_values = [f(x_init)]
    m0=0
    λ=0.99
    v0=0
    β=0.999
    for i in range(num_iterations):
        #取最后一个x值
        x = x_values[-1]
        #计算梯度
        gradient = np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)
        #计算新的M值
        M_new = λ * m0 - learning_rate * gradient
        #计算新的v值
        v_new = β * v0 ** 2 + (1-β) * gradient**2
        #计算新的x值
        x_new = x + M_new / (np.sqrt(v_new) + 1e-8)
        x_values.append(x_new)
        f_values.append(f(x_new))
    return x_values, f_values
     
     
# 绘制图形
def plot(x_values, f_values):
    plt.plot(x_values, f_values, '-o')
    plt.plot(x_init, f(x_init), '-')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(' x and f(x) Variation')
    plt.grid(True)
    plt.show()


# 设置初始值、学习率和迭代次数
x_init =np.linspace(-5, 5, 100)
x=-4
learning_rate = 0.4
num_iterations = 10


# 调用梯度下降函数获取x和f(x)的变化情况
x_values, f_values = gradient_descent(np.array([x]), learning_rate, num_iterations)
plot(x_values, f_values)
# 调用随机梯度下降函数获取x和f(x)的变化情况
x_values, f_values = stochastic_gradient_descent(np.array([x]), learning_rate, num_iterations)
plot(x_values, f_values)
# 调用AdaGrad函数获取x和f(x)的变化情况
x_values, f_values = adagrad(np.array([x]), learning_rate, num_iterations)
plot(x_values, f_values)
# 调用RMSProp函数获取x和f(x)的变化情况
x_values, f_values = RMSProp(np.array([x]), learning_rate, num_iterations)
plot(x_values, f_values)
# 调用Momentum函数获取x和f(x)的变化情况
x_values, f_values = Momentum(np.array([x]), learning_rate, num_iterations)
plot(x_values, f_values)
# 调用Adam函数获取x和f(x)的变化情况
x_values, f_values = Adam(np.array([x]), learning_rate, num_iterations)
plot(x_values, f_values)
