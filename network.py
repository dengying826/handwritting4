import random
import pickle
import gzip
import numpy as np

# 加载并处理数据
# 数据转换
# tr_d是由50000个长度为784的numpy.ndarray组成的tuple
# 转换后的training_inputs是由50000个长度为784的numpy.ndarray组成的list
def load_data(filepath):
    # 读取数据
    f = gzip.open(filepath, 'rb')
    tr_d, va_d, te_d = pickle.load(f, encoding='latin1')
    f.close()
    print(type(tr_d))
    # 训练集
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]  # 输入

    training_results = [one_hot(y) for y in tr_d[1]]  # 标签
    training_data = list(zip(training_inputs, training_results))

    # 测试集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    print(len(training_data),len(test_data))
    return (training_data, test_data)

# 独热编码
def one_hot(x):
    vec = np.zeros((10, 1))
    vec[x] = 1.0
    return vec

# 激活函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 激活函数的导数
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 交叉熵损失函数
class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        # return -(np.dot(y.transpose(), np.nan_to_num(np.log(a))) + np.dot((1 - y).transpose(), np.nan_to_num(np.log(1 - a))))
    @staticmethod
    def delta(a, y):
        return (a-y)


# 网络
class Network(object):

    # 初始化神经网络拓扑结构
    def __init__(self, sizes, cost=CrossEntropyCost):
        # size: [784, 30, 30, 10]  元素表示每层的神经元数量
        # cost：损失函数，默认使用交叉熵
        self.layers_num = len(sizes)
        self.sizes = sizes
        self.weight_initializer()
        self.cost=cost

    # 随机初始化权重、偏置矩阵
    def weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # 前向输入
    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            # 矩阵相乘
            z = np.dot(w, x) + b
            # 激活函数
            x = sigmoid(z)
        return x

    # 反向传播
    def backprop(self, x, y):
        # 占位
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 第一步：前向传播，求出每个神经元的输出项
        # 为什么反向传播过程中仍然需要forward, 因为我们需要在forward中记录每层z，activation变量，方便我们以后计算梯度。
        activation = x
        activations = [x]  # 分层存储每层的输出项（对应上文中的 a）
        # w*x = z => sigmoid => x/activation
        zs = []  # 分层存储每层的 z 向量（对应上文中的 z）

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 第二步：反向传播
        # 不同代价函数唯一的不同在于计算最后一层的$\delta^\ell$
        # self.cost 交叉熵
        # 2.1 首先计算输出层计算梯度。[10, 1] * [10, 1] => [10, 1]
        delta = (self.cost).delta(activations[-1], y)
        # 2.2 求偏置量的梯度
        nabla_b[-1] = delta
        # 2.3 求权重参数的梯度
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 2.4 反向传播，依次更新每层的每个神经元的权重和偏移量
        for layer in range(2, self.layers_num):
            # L = 1表示最后一层神经元， L = 2倒数第二层神经元
            z = zs[-layer]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return (nabla_b, nabla_w)


    # 准确率
    def accuracy(self, data, convert=False):
        # convert ：训练集True 测试集False
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    # 计算整个数据集上的代价
    def total_cost(self, data, lmbda, convert=False):
        # convert ：训练集False 测试集True
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = one_hot(y)
            cost += self.cost.fn(a, y)/len(data)

        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2
                                   for w in self.weights)
        return cost

    # 训练
    def train(self, training_data, test_data, epochs, batch_size, learning_rate,
            lmbda = 0.0,
            monitor_test_cost=False,
            monitor_test_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        # 如果有测试集长度
        n_data = len(test_data)
        # 训练集长度
        n = len(training_data)
        # # 训练损失、准确率, 测试损失、准确率
        training_cost, training_accuracy, test_cost, test_accuracy = [], [], [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_batch( mini_batch, learning_rate, lmbda, len(training_data))
            print("Epoch %s 训练完成：" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("训练数据损失: {}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                # print("训练数据准确率: {} / {}".format( accuracy, n))
                print("训练数据准确率: ",accuracy/n)

            if monitor_test_cost:
                cost = self.total_cost(test_data, lmbda, convert=True)
                test_cost.append(cost)
                print("测试数据损失: {}".format(cost))

            if monitor_test_accuracy:
                accuracy = self.accuracy(test_data)
                test_accuracy.append(accuracy)
                # print("测试数据准确率: {} / {}".format(self.accuracy(test_data), n_data))
                print("测试数据准确率:",self.accuracy(test_data)/n_data)

        # return test_cost, test_accuracy, training_cost, training_accuracy


    def update_batch(self, mini_batch, learning_rate, lmbda, n):
        # 占位
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 损失值
        # 当前批中每个样例
        for x, y in mini_batch:
            # 得到当前的梯度值
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 就比如是：[w1, w2, w3]这个是一个样本的，多样本的时候我们应该吧对应位置的累加起来求一个平均值。
            # cur当前的，accu为之前的；进行对应位置累加。
            nabla_b = [accu + cur for accu, cur in zip(nabla_b, delta_nabla_b)]
            nabla_w = [accu + cur for accu, cur in zip(nabla_w, delta_nabla_w)]
        # 求平均值梯度值w, b

        # 使用规范化学习规则 更新权值w偏置b
        self.weights = [(1-learning_rate*(lmbda/n))*w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]





if __name__ == '__main__':
    # 加载训练集与测试集
    training_data, test_data = load_data('data/mnist.pkl.gz')
    # 初始化网络
    # network = Network([784,30,30,10],cost = CrossEntropyCost)
    # 训练
    # network.train(training_data,test_data, 200, 32, 2,
    #           monitor_training_accuracy=True, monitor_test_accuracy=True)

