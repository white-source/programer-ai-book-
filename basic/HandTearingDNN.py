import random
# 1  训练集合 测试数据集
dataset = [[2.7834, 4.55667, 0],
[3.3778, 4.4424,0],
[1.345345, 1.45646, 0],
[7.62324, 2.75646, 1],
[5.333, 2.08, 1],
[6.9223, 1.77745, 1],
[3.0634, 3.0005, 0]]

test_data = [[1.45623452, 2.3545, 0],
[8.86356, -0.23434, 1],
[7.67356, 3.053434, 1]]

# 2 定义神经网络
def initial_network(n_inputs, n_hidden, n_outputs):
    network= list()
    hidden_layer= [{'weight': [random.random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    output_layer= [{'weight': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]

    network.append(hidden_layer)
    network.append(output_layer)
    return network

print([{'weight': [random.random() for i in range(5)]} for i in range(2)])
