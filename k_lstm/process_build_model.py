import sys
print('Python %s on %s' % (sys.version, sys.platform))
#sys.path.extend(['C:/git_PycharmProjects/time-series-forecasting/k_lstm'])

import os
#os.chdir('C:/git_PycharmProjects/time-series-forecasting/k_lstm')
#print(os.getcwd())

from importlib import reload

import k_lstm.load_data_temperature as load_data_temperature

import k_lstm.prepare_data as prepare_data
reload(prepare_data)

import k_lstm.training as training


#--------- Configuration ------------

# input data
data_path = '\\..\\data_input\\'
data_path = os.getcwd() + data_path
print("data_path: " + data_path)
ts_sample_frequency = '60min'  # original
ts_sample_frequency = 'D'

# temp data folder
temp_data_folder = '\\data_temp\\'
temp_data_folder = os.getcwd() + temp_data_folder
print(temp_data_folder)

# training data configuration
look_back = 15  # recent history for training
look_forward = 10
ts_features = ['month']
train_size_rate = 0.7

# neuron network configuration
num_layers = 2
num_neurons = 10
num_epochs = 100

# model configuration
step_range = [1, 2]  # must between  1 to look_forward, for one_step_model

#------------------------------------

ts_df = load_data_temperature.get_ts(data_path)

prepare_data.prepare_data(ts_df, ts_sample_frequency, temp_data_folder, look_back, look_forward, train_size_rate, ts_features)

train_x, train_y = training.get_train_data(temp_data_folder)

training.build_mutliple_steps_model(train_x, train_y, temp_data_folder, num_layers, num_neurons, num_epochs, ts_features)

training.build_one_step_model(train_x, train_y, temp_data_folder, num_layers, num_neurons, num_epochs, step_range)

