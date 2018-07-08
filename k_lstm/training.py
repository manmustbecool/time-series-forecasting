# pip install --upgrade tensorflow==1.5 for old CPU

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from importlib import reload

import numpy as np

import math
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import datetime

import k_lstm.k_utils as my_keras_utils

import k_lstm.k_config as k_config
reload(k_config)
temp_data_folder = k_config.temp_data_folder


import k_lstm.my_utils as my_utils
reload(my_utils)
train_x = my_utils.get_object(temp_data_folder+'train_x.pkl')
train_y = my_utils.get_object(temp_data_folder+'train_y.pkl')

print(train_x.shape)
print(train_y.shape)

# create and fit the LSTM network
# ---- build a model --------
def build_model(train_x, train_y):

    # ---- model configuration ----
    # A batch size of 1 means that the model will be fit using online training (as opposed to batch training or mini-batch training).
    # As a result, it is expected that the model fit will have some variance.
    batch_size = 1
    timesteps = 1

    num_neurons = 10
    dropout = 0.05
    # Using all your batches once is 1 epoch.
    # Ideally, more training epochs would be used (such as 1500), but this was truncated to 50 to keep run times reasonable.
    epochs = 50

    num_in = train_x.shape[1]
    print('num_in:', num_in)
    num_out = train_y.shape[1]
    print('num_out:', num_out)


    # batch_input_shape = (batch_size, timesteps, data_dim).
    # 2D array [samples, features] to a 3D array [samples, timesteps, features].
    train_x = train_x.reshape(train_x.shape[0], timesteps, num_in)
    # train_y = train_y.reshape(train_y.shape[0], timesteps, num_out)

    # design network
    model = Sequential()
    model.add(
        LSTM(num_neurons, batch_input_shape=(batch_size, timesteps, num_in), stateful=True, return_sequences=True, dropout=dropout))
    model.add(LSTM(30, batch_input_shape=(batch_size, timesteps, num_in), stateful=True))
    model.add(Dense(num_out, activation='linear'))
    # using the efficient ADAM optimization algorithm and the mean squared error loss function
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

    # fit network
    fit_history = dict(loss=[], mse=[], mae=[], mape=[], title='')  # Creating a empty list for holding the loss later
    for i in range(epochs):
        print('-- Epochs -- ' + str(i))
        result = model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        # The LSTM is stateful;  manually reset the state of the network at the end of each training epoch
        model.reset_states()  # clears only the hidden states of your network

        # save fit errors
        fit_history['loss'].append(result.history['loss'])  # Now append the loss after the training to the list.
        fit_history['mse'].append(result.history['mean_squared_error'])
        fit_history['mae'].append(result.history['mean_absolute_error'])
        fit_history['mape'].append(result.history['mean_absolute_percentage_error'])

    fit_history['title'] = "num_in:" + str(num_in) + 'num_out:' + str(num_out) + " num_neurons:" + str(num_neurons)
    # print(fit_history)
    print("Training completed")

    return model, fit_history

def plot_fit_history(fit_history, step=''):
    plt.figure()
    plot_fit_error(fit_history)
    plt.title(fit_history['title'])
    save_fit_plot(step)

def plot_fit_error(fit_history, label=''):
    # plt.plot(fit_history['loss'], label='loss')
    plt.plot(fit_history['mse'], label='mse' + label)
    # plt.plot(fit_history['mae'], label='mae')
    # plt.plot(fit_history['mape'], label='mape')


def save_fit_plot(name):
    plt.legend()
    plt.savefig(temp_data_folder+"fit_history "+str(datetime.datetime.now().strftime("%Y-%m-%d %H %M"))+" "+name+".png")



# build a full model
# build model
model, fit_history = build_model(train_x, train_y)
# plot model fit history
plot_fit_history(fit_history)
# save model
my_keras_utils.save_model(model, temp_data_folder)


num_output = train_y.shape[1]
plt.figure()
for step in range(0, num_output):
    train_y_step = train_y[:, step]
    train_y_step = train_y_step.reshape(train_y.shape[0], 1)
    model, fit_history = build_model(train_x, train_y_step)
    # plot model fit history
    step_text = ' step_' + str(step + 1)

    my_keras_utils.save_model(model, temp_data_folder + 'step_' + str(step + 1) + '_')

    plot_fit_error(fit_history, step_text)

plt.legend()
plt.title(fit_history['title'])
save_fit_plot(step_text)



#--------------------------------------------


num_in = train_x.shape[1]
predicted_y = np.array([])
for ix in range(len(train_x)):
    predicted = model.predict(train_x[ix].reshape(1, 1, num_in))
    predicted_y = np.append(predicted_y, predicted)

predicted_y = predicted_y.reshape(train_y.shape[0], train_y.shape[1])

rmse = math.sqrt(mean_squared_error(train_y, predicted_y))
print('Test RMSE: %.3f' % rmse)
