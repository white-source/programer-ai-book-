from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

def createModel():
	model = Sequential()
	model.add(Dense(32, input_shape=(13,), activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
model = createModel()
model.fit(x_train, y_train, batch_size=8, epochs=10000)

# print(model.metrics_names)
# print(model.evaluate(x_test, y_test))

for i in range(10):
	y_pred = model.predict([[x_test[i]]])
	print("predict: {}, target: {}".format(y_pred[0][0], y_test[i]))

#27.440169689702053


# Epoch 9999/10000
# 404/404 [==============================] - 0s 64us/sample - loss: 1.1444
# Epoch 10000/10000
# 404/404 [==============================] - 0s 66us/sample - loss: 0.9892
# ['loss']
# 102/102 [==============================] - 0s 176us/sample - loss: 20.1909
# 20.19088176652497
