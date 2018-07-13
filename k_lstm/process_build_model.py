from importlib import reload

import k_lstm.load_data_temperature as data_input

import k_lstm.prepare_data as prepare_data
reload(prepare_data)

import k_lstm.training as training

import os

temp_data_folder = '\\k_lstm\\data_temp\\'
temp_data_folder = os.getcwd() + temp_data_folder
print(temp_data_folder)

data_path = '\\k_lstm\\data_input\\'
data_path = os.getcwd() + data_path

look_back = 20
look_forward = 10

ts_features = ['month']

train_size_rate = 0.7

num_layers = 2
num_neurons = 30
num_epochs = 5
# step_range = [1]  # must between  1 to look_forward


ts_df, ts_sample_frequency = data_input.get_ts(data_path)

prepare_data.prepare_data(ts_df, ts_sample_frequency, temp_data_folder, look_back, look_forward, train_size_rate, ts_features)

train_x, train_y = training.get_train_data(temp_data_folder)

training.build_mutliple_steps_model(train_x, train_y, temp_data_folder, num_layers, num_neurons, num_epochs, ts_features)

# training.build_one_step_model(train_x, train_y, temp_data_folder, num_layers, num_neurons, num_epochs, step_range)

