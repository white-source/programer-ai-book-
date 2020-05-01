from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Masking, Lambda, Flatten, Reshape, Permute, RepeatVector, Activation
from tensorflow.keras.datasets import boston_housing
import numpy as np

def BaselineMode():
	model = Sequential()
	model.add(Dense(13, input_shape=(13,), activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def DenseMode():
	model = Sequential()
	model.add(Dense(64, input_shape=(13,), activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# baseModel = BaselineMode()
# baseModel.fit(x_train, y_train, batch_size=5, epochs=100)

# print(baseModel.metrics_names)
# print(baseModel.evaluate(x_test, y_test))


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(4, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, y, batch_size=1, epochs=10000)
print(model.predict(X))

# [[0.49929813]
#  [0.49999362]
#  [0.49997586]
#  [0.5006713 ]]

[[0.0084908 ]
 [0.9729198 ]
 [0.97606194]
 [0.03987955]]

# error_count = 0
# for i in range(len(x_test)):
# 	x = np.array([x_test[i]])
# 	y_pred = baseModel.predict(x)
# 	print("{}, {}".format(y_pred[0][0], y_test[i]))
# 	if y_pred != y_test[i]:
# 		error_count+=1

# print('errors: {}, accuracy: {}'.format(error_count, error_count/len(x_test)))





