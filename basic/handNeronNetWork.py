#1 损失函数
def cost_function(self, X, Y, weight, bias):
    n = len(X)
    total_error = 0.0
    for i in range(n):
        total_error += (Y[i] - (weight * X[i] + bias))**2
    return total_error/n


#2 数学推导


#3 梯度下降 更新权重
def update_weights(self, X, Y, weight, bias, learning_rate):
    dw = 0
    db = 0
    n = len(X)
    for i in range(n):
        dw += -2 * X[i](Y[i] - (weight * X[i] + bias))
        db += -2 * (Y[i] - (weight * X[i] + bias))

    weight -= (dw / n) * learning_rate
    bias -= (db / n) * learning_rate

    return weight,bias

# 4  归一化
#Xi= (Xi - Xmin) / (Xmax - Xmin)





