from importlib import reload

import k_lstm.load_data_temperature as data_input

import k_lstm.prepare_data as prepare_data
reload(prepare_data)

import k_lstm.training as training

import os


temp_data_folder = '\\..\\k_tree\\data_temp\\'
temp_data_folder = os.getcwd() + temp_data_folder
print(temp_data_folder)


# input data
data_path = '\\..\\k_lstm\\data_input\\'
data_path = os.getcwd() + data_path
ts_sample_frequency = '60min'  # original
ts_sample_frequency = 'D'


look_back = 15
look_forward = 50

ts_features = ['month']

train_size_rate = 0.7

step_range = [1]  # must between  1 to look_forward, for one_step_model


ts_df = data_input.get_ts(data_path)

prepare_data.prepare_data(ts_df, ts_sample_frequency, temp_data_folder, look_back, look_forward, train_size_rate, ts_features)

train_x, train_y = training.get_train_data(temp_data_folder)

from sklearn.tree import DecisionTreeRegressor


model = DecisionTreeRegressor(max_depth=5)
model.fit(train_x, train_y)

# save model
import k_lstm.my_utils as my_utils
reload(my_utils)

my_utils.save_object(model, temp_data_folder+"model")


# training.build_mutliple_steps_model(train_x, train_y, temp_data_folder, num_layers, num_neurons, num_epochs, ts_features)

# training.build_one_step_model(train_x, train_y, temp_data_folder, num_layers, num_neurons, num_epochs, step_range)

import k_lstm.my_utils as my_utils
model_tree = my_utils.get_object(temp_data_folder+"\\model.pkl")

y_1 = model_tree.predict(train_x)

print(y_1)