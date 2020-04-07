import random
import math

# 1  训练集合 测试数据集
dataset = [[2.7834, 4.55667, 0],
           [3.3778, 4.4424, 0],
           [1.345345, 1.45646, 0],
           [7.62324, 2.75646, 1],
           [5.333, 2.08, 1],
           [6.9223, 1.77745, 1],
           [3.0634, 3.0005, 0]]

test_data = [[1.45623452, 2.3545, 0],
             [8.86356, -0.23434, 1],
             [7.67356, 3.053434, 1]]

n_inputs = 2
n_outputs = 2


# 2 定义神经网络
# 一个全连接层隐层的权重参数 总个数 = (n_inputs+1) * n_hidden
# output层 权重参数 总个数 = (n_hidden+1) * n_outputs


def initial_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]}
                    for i in range(n_hidden)]
    output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]}
                    for i in range(n_outputs)]
    network.append(hidden_layer)
    network.append(output_layer)
    return network


def initial_network_mutiply(n_inputs, n_hidden, n_outputs):
    network = list()

    hidden_layer1 = [{'weights': [random.random() for i in range(n_inputs + 1)]}
                     for i in range(n_hidden)]

    hidden_layer2 = [{'weights': [random.random() for i in range(n_hidden + 1)]}
                     for i in range(n_hidden)]

    output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]}
                    for i in range(n_outputs)]
    network.append(hidden_layer1)
    network.append(hidden_layer2)
    network.append(output_layer)
    return network


# 数据demo
# [
#     {'weight': [0.8151781039319206, 0.6860059592681256,0.60059592681256]},
#     {'weight': [0.19909599477087803, 0.6639829849695144,0.11059592681256]}
# ]
# print([{'weight': [random.random() for i in range(5)]} for i in range(2)])

# A 任何模型的训练-其关键都是实现 4个核心函数：
# A-1 net_input :计算神经元的网络输入
# A-2 activation:激活函数，将神经元的网络输入映射到下一层的输入空间
# A-3 cost-function:计算损失函数
# A-4 update_weights:更新权重

# B 反向传播：
# B-1 ->对网络中所有权重都计算损失函数的梯度
# B-2 -> 这个梯度在优化算法中，用来更新权重，以最小化损失函数
# B-3 ->（实际上：他指代所有基于梯度下降利用连式法则，来训练神经网络的算法，以帮助实现可递归循环的形式，来有效每一层的权重更新，直到获得期望的效果）

# 3 每个神经元的网络输入


def net_input(weight, input):
    total_input = weight[-1]
    for i in range(len(weight) - 1):
        total_input += weight[i] * input[i]
    return total_input


# 4 激活函数
# 作用：用于将网络输入映射到区间(-1,1)的函数


def activation(total_input):
    return 1.0 / (1.0 + math.exp(-total_input))


# 5 前向传播的实现


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        outputs = []
        for neuron in layer:
            total_input = net_input(neuron['weights'], inputs)
            neuron['output'] = activation(total_input)
            outputs.append(neuron['output'])
        inputs = outputs
    return inputs


# 6 损失函数 ,损失函数不是必须的，根据 cost-function函数进行求导才是必须的


def cost_function(expected, outputs):
    n = len(expected)
    total_error = 0.0
    for i in range(n):
        total_error += (expected[i] - outputs[i]) ** 2
    return total_error


# 7 sigmoid激活函数的导数实现


def transfer_derative(output):
    return output * (1.0 - output)


# 8 有了6，7 最重要的方向传播实现
def backward_propagate(network, expected):
    # network 实际上是一个list,每个item都是一组参数，其中包含了 权重和其他属性；expected 包含结果的真实分类
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i == len(network) - 1:
            for j in range(len(layer)):
                neuron = layer[j]
                error = -2 * (expected[j] - neuron['output'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)

        # 将不同层的 误差 反向传递 回去
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derative(neuron['output'])


# 9 更新权重
def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= learning_rate * \
                                        neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= learning_rate * neuron['delta']


# TODO
# 10 所有核心的反向传播权重调整所用的函数OK，开始训练
def train_network(network, training_data, learning_rate, n_epoch, n_outputs):
    global epoch, sum_error
    for epoch in range(n_epoch):
        sum_error = 0
        for row in training_data:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += cost_function(expected, outputs)
            backward_propagate(network, expected)
            update_weights(network, row, learning_rate)
    print('>epoch:%d, learning rate: %.3f,error:%.3f' %
          (epoch, learning_rate, sum_error))


# 11 预测 ，把概率最大的分类选出来
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# start go
network = initial_network_mutiply(n_inputs, 1, n_outputs)
train_network(network, training_data=dataset, learning_rate=0.5, n_epoch=3000, n_outputs=n_outputs)

for row in test_data:
    result = predict(network, row)
    print('expected:%d,predicted:%d\n' % (row[-1], result))
