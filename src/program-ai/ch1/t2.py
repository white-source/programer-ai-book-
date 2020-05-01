import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=2))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='sgd')

x = np.array([[5,2],[4,2],[3,3]])
y = np.array([[30],[20],[60]])
model.fit(x, y, epochs=1000, batch_size=1)

test_x = np.array([[5,2],[4,2],[3,3]])
test_y = model.predict(test_x)

for i in range(len(test_x)):
	print(test_y[i])
	# print('input {} => predict: {}'.format(test_x[i], test_y[i]))
