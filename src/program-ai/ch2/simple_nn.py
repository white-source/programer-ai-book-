from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	# weights number contains an extra one for input theta0 which is always 1 (according to coursera ml course)
	hidden_layer1 = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	hidden_layer2 = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
	output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(hidden_layer1)
	network.append(hidden_layer2)
	network.append(output_layer)
	return network

def net_input(weights, inputs):
	# a = sum(w * input.T) + bias
	total_input = weights[-1]
	for i in range(len(weights)-1):
		total_input += weights[i] * inputs[i]
	return total_input

def activation(total_input):
	# sigmoid or reLu or whatever
	return 1.0/ (1.0 + exp(-total_input))

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

def transfer_derivative(output):
	# d(sigmoid(z)) = z * (1-z), standard bp algorithm
	return output * (1.0 - output)


def cost_function(expected, outputs):
	n = len(expected)
	total_error = 0.0
	for i in range(n):
		total_error += (expected[i] - outputs[i])**2
	return total_error


def backward_propagate(network, expected):
	# error[i] = W * error[i+1] * sigmoid_derivative
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()

		if i == len(network) - 1:
			# first from output layer, just get the output difference
			# note: our difference is expected - output, some other implementing uses output-expected
			#       which causes the weight update is different (weight = weight +/- ....)
			for j in range(len(layer)):
				neuron = layer[j]
				error = -2 * (expected[j] - neuron['output'])
				errors.append(error)
				# errors.append(expected[j] - neuron['output'])
		else:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i+1]:
					# neuron['delta'] saves errors[i+1] * sigmoid_derivateive
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)

		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, learning_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			# for hidden layer and output layer
			inputs = [neuron['output'] for neuron in network[i-1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= learning_rate * neuron['delta'] * inputs[j]
			# theta0 is always 1 (explained on coursera ml course)
			neuron['weights'][-1] -= learning_rate * neuron['delta']

def train_network(network, training_data, learning_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in training_data:
			outputs = forward_propagate(network, row)
			# this is a trick to set the bi-class flag by using the row[-1] as class index
			# expected[0] = 1: expected to be class 1
			# expected[1] = 1: expected to be class 2
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			# sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
			sum_error += cost_function(expected, outputs)
			backward_propagate(network, expected)
			update_weights(network, row, learning_rate)
		print('>epoch: %d, learning rate: %.3f, error: %.3f' % (epoch, learning_rate, sum_error))


def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# seed(1)
dataset = [[2.7810836,4.550537003,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1]]

test_data = [[1.465489372,2.362125076,0],
			[8.675418651,-0.242068655,1],
			[7.673756466,3.508563011,1]]

n_inputs = 2
n_outputs = 2
network = initialize_network(n_inputs, 1, n_outputs)
train_network(network, training_data = dataset, learning_rate = 0.5, n_epoch = 2000, n_outputs = n_outputs)
# for layer in network:
# 	print(layer)

for row in test_data:
	result = predict(network, row)
	print('expected: %d, predicted: %d\n' % (row[-1], result))