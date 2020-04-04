import random
class LinearRegression(object):
    # 核心思想：
    # 1 用模型去拟合数据，如果数据有偏差，那么拟合的模型也有偏差
    # 2 最大的困难是转换编程思维方式：从传统的面向具体逻辑流程的实现 变为 面向数据和结果拟合的实现 
    # 3 其次概念 和原理的理解 
    def __init__(self, eta=0.01, iterations=10):
        self.lr = eta
        self.iterations = iterations
        self.w = 0.0
        self.bias = 0.0

    # 1 目的是干什么，对于整个训练起到什么作用，在整个链路中 关系，承上启下的是哪两个 2 输入输出是什么，3 逻辑结构是什么 4 和更新权重有什么关系
    def cost_function(self, X, Y, weight, bias):
        n = len(X)
        total_error = 0.0
        # 目的：循环 每个样本，计算 样本和预测值的误差，并取平均,逻辑：样本和预测值的差就是误差，再对误差平方
        # 输入 是 样本+标签；权重+偏置  输出是 误差和的平均
        for i in range(n):
            total_error += (Y[i] - (weight * X[i] + bias))**2
        return total_error / n

    # 目的 是什么，在整个 链路中承上启下的部分
    def update_weights_all(self, X, Y, weight, bias, learning_rate):
        dw = 0
        db = 0
        n = len(X)
        for i in range(n):
            dw += -2 * X[i] * (Y[i] - (weight * X[i] + bias))
            db += -2 * (Y[i] - (weight * X[i] + bias))
        weight -= (dw / n) * learning_rate
        bias -= (dw / n) * learning_rate
        return weight,bias

     # 1 上面的更新权重方法，每次都载入全量训练数据 训练
     # 2 每次只用一条随机数据来训练-》这就是随机梯度下降的原理，弊端 会让训练时间变长
     # 3 minbatch每次随机选择，总训练数据的一个子集（和模型本身，和数据特点有关系）
     # batch_size =1 ,就是随机梯度下降
    def update_weights(self, X, Y, weight, bias, learning_rate):
        dw = 0
        db = 0
        n = len(X)
        indexes  = list(range(n))
        random.shuffle(indexes)
        batch_size = 4
        for k in range(batch_size):
            i = indexes[k]
            dw += -2 * X[i] * (Y[i] - (weight * X[i] + bias))
            db += -2 * (Y[i] - (weight * X[i] + bias))
        weight -= (dw / n) * learning_rate
        bias -= (dw / n) * learning_rate
        return weight,bias



    # 训练拟合数据
    def fit(self, X, Y):
        cost_history = []
        # 1 多轮训练迭代 Ps:大脑就行 森林里的一块石头，等你慢下来，静下来，石头上就沁出水来了，而这 一轮一轮的水，给予人 生命和认知的动力，继续前行
        for i in range(self.iterations):
            self.w, self.bias = self.update_weights(
                X, Y, self.w, self.bias, self.lr)
            # 2 计算误差，用于观察和监控训练过程
            cost = self.cost_function(X, Y, self.w, self.bias)
            cost_history.append(cost)

            if i % 10 == 0:
                print("iter = {:d} weight={:.2f} bias={:.4f} cost={:.2f}".format(
                    i, self.w, self.bias, cost))

        return self.w, self.bias, cost_history

    # 预测
    def predict(self, x):
        x = self.nomalization(x, 100, -100)
        return self.w * x + self.bias

    # 数据归一化
    def nomalization(self, x, max, min):
        return (x - min) / (max - min)

x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
model = LinearRegression(0.01, 500)

# 归一化数据,使所有数据都落在（0，1）区间内
X = [model.nomalization(k, 100, -100) for k in x]
model.fit(X, y)
test_x = [90, 80, 81, 82, 75, 40, 32, 15, 5,
          1, -1, -15, -20, -22, -33, -45, -60, -90]
for i in range(len(test_x)):
    print('input {} => predict:{}'.format(test_x[i], model.predict(test_x[i])))

#这里是利用梯度下降来修正权重，达到不断逼近最优解的目的

