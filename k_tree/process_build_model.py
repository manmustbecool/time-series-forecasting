from importlib import reload

import k_lstm.load_data_temperature as data_input

import k_lstm.prepare_data as prepare_data
reload(prepare_data)

import k_lstm.training as training

import os


import k_lstm.my_utils as my_utils
reload(my_utils)

#--------- Configuration ------------

temp_data_folder = '\\data_temp\\'
temp_data_folder = os.getcwd() + temp_data_folder
print("data_temp: "+temp_data_folder)

# input data
data_path = '\\..\\data_input\\'
data_path = os.getcwd() + data_path
ts_sample_frequency = '60min'  # original
ts_sample_frequency = 'D'

# training data
look_back = 15
look_forward = 50
ts_features = ['month']
train_size_rate = 0.7

# tree configuration
tree_max_depth = 5

# model configuration
step_range = [1, 2]  # must between  1 to look_forward, for one_step_model

#---------------------------------

ts_df = data_input.get_ts(data_path)

prepare_data.prepare_data(ts_df, ts_sample_frequency, temp_data_folder, look_back, look_forward, train_size_rate, ts_features)

train_x, train_y = training.get_train_data(temp_data_folder)

from sklearn.tree import DecisionTreeRegressor


def build_mutliple_steps_model(train_x, train_y, temp_data_folder, tree_max_depth):
    model = DecisionTreeRegressor(max_depth=tree_max_depth)
    model.fit(train_x, train_y)

    # save model
    my_utils.save_object(model, temp_data_folder + "model")


def build_one_step_model(train_x, train_y, temp_data_folder, tree_max_depth, step_range):
    if step_range is None:
        num_output = train_y.shape[1]
        step_range = range(0, num_output)

    for step in step_range:
        step_text = ' step_' + str(step)
        print("build model for " + step_text)

        train_y_step = train_y[:, (step - 1)]
        train_y_step = train_y_step.reshape(train_y.shape[0], 1)

        model = DecisionTreeRegressor(max_depth=tree_max_depth)
        model.fit(train_x, train_y_step)

        # save model
        my_utils.save_object(model, temp_data_folder + 'step_' + str(step) + '_' + "model")


build_mutliple_steps_model(train_x, train_y, temp_data_folder, tree_max_depth)

build_one_step_model(train_x, train_y, temp_data_folder, tree_max_depth, step_range)


#---------- test saved model -----------

import k_lstm.my_utils as my_utils
model_tree = my_utils.get_object(temp_data_folder+"\\model.pkl")
y_1 = model_tree.predict(train_x)
print("test result:" + str(y_1))