import k_lstm.load_data_temperature as load_data_temperature

import k_lstm.prepare_data as prepare_data

import k_lstm.training as training

temp_data_folder = 'data_temp/'

look_back = 10
look_forward = 5
train_size_rate = 0.7

num_neurons = 10
num_epochs = 20
step_range = [4]  # must between  1 to look_forward


ts_df = load_data_temperature.ts_df
ts_df_frequency = load_data_temperature.ts_df_frequency

prepare_data.prepare_data(ts_df, ts_df_frequency, temp_data_folder, look_back, look_forward, train_size_rate)

train_x, train_y = training.get_train_data(temp_data_folder)

training.build_mutliple_steps_model(train_x, train_y, temp_data_folder, num_neurons, num_epochs)

training.build_one_step_model(train_x, train_y, temp_data_folder, num_neurons, num_epochs, step_range)
