import matplotlib.pyplot as plt
import math
import numpy

def sigmoid(x):
	return 1/(1+math.exp(-x))


def elu(x, alpha=1.0):
	return x if x > 0 else alpha*(math.exp(x)-1)


def selu(x):
	scale = 1.0507
	return scale * elu(x)


def relu(x, alpha=0.0, max_value=None, threshold=0.0):
	return max(x, 0)


def softplus(x):
	return math.log(math.exp(x) + 1)


def softsign(x):
	return x / (abs(x) + 1)


def tanh(x):
	return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def hard_sigmoid(x):
	if x < -2.5:
		return 0
	if x > 2.5:
		return 1
	return 0.2 * x + 0.5


def exponential(x):
	return math.exp(x)


def linear(x):
	return x


x=[k for k in numpy.arange(-10.0, 10.0, 0.1)]

activations = [sigmoid, elu, selu, relu, softplus, softsign, tanh, hard_sigmoid, exponential, linear]

i = 1
for f in activations:
	activation_name = f.__code__.co_name
	plt.subplot(len(activations)/5 + 1,5,i)
	plt.plot(x, [f(k) for k in x])
	plt.title(activation_name)
	i+=1

plt.show()

# plt.subplot(3,4,2)
# plt.plot(x, [elu(k) for k in x])
# plt.title('elu')

# plt.subplot(3,4,3)
# plt.plot(x, [selu(k) for k in x])
# plt.title('selu')

# plt.subplot(3,4,4)
# plt.plot(x, [relu(k) for k in x])
# plt.title('relu')


