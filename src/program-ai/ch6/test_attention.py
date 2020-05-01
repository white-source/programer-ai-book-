from numpy import array
from numpy import argmax
from numpy import array_equal
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Input, Dense, RepeatVector, Flatten
from tensorflow.keras.layers import Activation, Permute, multiply

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]
 
# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
 
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

def get_pair(n_in, n_out, cardinality):
	# generate random sequence
	sequence_in = generate_sequence(n_in, cardinality)

	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
	# one hot encode
	X = one_hot_encode(sequence_in, cardinality)
	y = one_hot_encode(sequence_out, cardinality)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y


def attention_model(n_timesteps_in, n_features):
       units = 50
       inputs = Input(shape=(n_timesteps_in, n_features))

       encoder = LSTM(units, return_sequences=True, return_state=True)
       encoder_outputs, encoder_states, _ = encoder(inputs)

       a = Dense(1, activation='tanh', bias_initializer='zeros')(encoder_outputs)
       a = Flatten()(a)
       annotation = Activation('softmax')(a)
       annotation = RepeatVector(units)(annotation)
       annotation = Permute((2, 1))(annotation)

       context = multiply([encoder_outputs, annotation])
       output = Dense(n_features, activation='softmax', name='final_dense')(context)

       model = Model([inputs], output)	
       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
       return model

def train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features):
       for epoch in range(5000):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	model.fit(X, y, epochs=1, verbose=0)

       total, correct = 100, 0
       for _ in range(total):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	yhat = model.predict(X, verbose=0)
	result = one_hot_decode(yhat[0])
	expected = one_hot_decode(y[0])
	if array_equal(expected, result):
       	    correct += 1

       return float(correct)/float(total)*100.0


n_features = 50
n_timesteps_in = 6
n_timesteps_out = 3
n_repeats = 5

for _ in range(n_repeats):
      model = attention_model(n_timesteps_in, n_features)
      accuracy = train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features)
      print(accuracy)

