import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import plot_model

import json

config_file = './s2s_1_config.json'
model_file = './s2s_1.h5'

# configure
config = {}
with open(config_file) as f:
	config = json.load(f)

num_encoder_tokens = config['num_encoder_tokens']
num_decoder_tokens = config['num_decoder_tokens']
latent_dim = config['latent_dim']
max_num_samples = config['max_num_samples']

model = load_model(model_file)

encoder_inputs = model.layers[0].input   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.layers[1].input   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model( [decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)


input_token_index = config['input_token_index']
target_token_index = config['target_token_index']
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq, max_decoder_seq_length):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence



data_path = './chat.txt'

input_texts = []
input_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[:len(lines) - 1]:
    input_text, target_text = line.split('\t')
    input_texts.append(input_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)

input_characters = sorted(list(input_characters))
num_encoder_tokens = len(input_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])

def test(input_text):
    input_data = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    for t, char in enumerate(input_text):
        input_data[0, t, input_token_index[char]] = 1.    

    response = decode_sequence(input_data, config['max_decoder_seq_length'])
                 print('input:{}, response:{}'.format(input_text, response))

test_data = [
            'hello',
            'hello world',
            'how are you',
            'good morning',
            'cheers',
            'enjoy',
            ]

for _, text in enumerate(test_data):
    test(text)
