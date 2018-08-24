import numpy as np

import math
from sklearn.metrics import mean_squared_error

from importlib import reload

import k_prophet.prophet as prophet
reload(prophet)

plt_step_ahead = 60
plt_max_back = 2000
plt_pause = 0.1

plt_cross_step_ahead = 5
plt_cross_threshold = 300
plt_warn_threshold = 10

ts_frequency = 7



import k_lstm.my_utils as my_utils
reload(my_utils)
ml_data_x = my_utils.get_object('k_lstm/data_temp/ml_data_x.pkl')
ts_df = my_utils.get_object('k_lstm/data_temp/ts_df.pkl')

ts_df_scaler = my_utils.get_object('k_lstm/data_temp/ts_df_scaler.pkl')
ts_features = my_utils.get_object('k_lstm/data_temp/ts_features.pkl')
ts_sample_frequency = my_utils.get_object('k_lstm/data_temp/ts_sample_frequency.pkl')

import matplotlib.pyplot as plt

import k_lstm.k_utils as k_utils
model_lstm = k_utils.load_model('k_lstm/data_temp/step_1_')
model_lstm = k_utils.load_model('k_lstm/data_temp/')



# ---- make predictions -------


def new_plt_data():
    x = dict(plt_value_index=np.array([]), plt_value=np.array([]), plt_past_predict=np.array([]), plt_past_predict_index=np.array([]),
             plt_ahead_predict=np.array([]), plt_ahead_predict_index=np.array([]), plt_cross_index=np.array([]), plt_cross=np.array([]),
             pltWarns=np.array([]), pltWarnIndex=np.array([]), pltErrorIndex=np.array([]), pltErrors=np.array([]))
    return x

plt.ion() ## Note this correction

def get_plt_data(ix, actual_value, multi_steps_predictions, pltData):

    # ix = 0;

    print("ix:", ix)

    # ---- plt_value -------
    pltData['plt_value'] = np.append(pltData['plt_value'], actual_value)
    if len(pltData['plt_value']) > plt_max_back:
        pltData['plt_value'] = np.delete(pltData['plt_value'], 0)

    pltData['plt_value_index'] = np.arange((ix + 1 - len(pltData['plt_value'])), (ix + 1))

    # print(pltData['plt_value'])
    # print(pltData['plt_value_index'])

    # ---- plt_past_predict -------
    temp = pltData['plt_past_predict']
    temp = np.append(temp, multi_steps_predictions[(plt_cross_step_ahead - 1)])
    if len(temp) > (plt_max_back + plt_cross_step_ahead):
        temp = np.delete(temp, 0)
    pltData['plt_past_predict'] = temp

    pltData['plt_past_predict_index'] = np.arange((ix + plt_cross_step_ahead + 1 - len(temp)), (ix + plt_cross_step_ahead + 1))
    # print('plotPredicts:', plt_past_predict)
    # print(plt_past_predict_index)

    # ---- plt_ahead_predict -------

    pltData['plt_ahead_predict'] = multi_steps_predictions.flatten()
    pltData['plt_ahead_predict_index'] = np.arange((ix + 1), (ix + 1 + plt_step_ahead))
   # else:
   #     pltData['plt_ahead_predict'] = np.append(np.array(pltData['plt_past_predict'][-2]), multiStepPredicts.flatten())
   #     pltData['plt_ahead_predict_index'] = np.arange((ix + 1 - 1), (ix + 1 + plotStepAhead))

    pltData['plt_ahead_predict_index'] = np.array(pltData['plt_ahead_predict_index'])

    # ---- plt_cross -------
    crossIndex = (pltData['plt_ahead_predict'] > plt_cross_threshold)
    crossIndex = np.append(np.full(plt_cross_step_ahead, False), crossIndex[plt_cross_step_ahead:])

    crosses = pltData['plt_ahead_predict'][crossIndex]
    crossIndex = pltData['plt_ahead_predict_index'][crossIndex]

    pltData['plt_cross'] = np.append(pltData['plt_cross'], crosses)
    pltData['plt_cross_index'] = np.append(pltData['plt_cross_index'], crossIndex)

    # clean old cross data
    pltData['plt_cross'] = pltData['plt_cross'][pltData['plt_cross_index'] >= pltData['plt_past_predict_index'][0]]
    pltData['plt_cross_index'] = pltData['plt_cross_index'][pltData['plt_cross_index'] >= pltData['plt_past_predict_index'][0]]

    # calculate warns
    crossIndexUnique = np.asarray(np.unique(pltData['plt_cross_index'], return_counts=True)).T
    crossIndexUnique = crossIndexUnique[crossIndexUnique[:, 1] > plt_warn_threshold]
    pltData['pltWarns'] = np.full(len(crossIndexUnique), plt_cross_threshold)
    pltData['pltWarnIndex'] = crossIndexUnique[:, 0]

    # ----  calculate  error ----------------
    minI = max(min(pltData['plt_past_predict_index']), min(pltData['plt_value_index']))
    maxI = min(max(pltData['plt_past_predict_index']), max(pltData['plt_value_index']))
    pltData['pltErrorIndex'] = np.arange(minI, maxI)
    # print('pltErrorIndex', pltData['pltErrorIndex'])
    rmse = 0
    if len(pltData['pltErrorIndex']) > 2:
        ground_truth = pltData['plt_value'][np.isin(pltData['plt_value_index'], range(minI, maxI))]
        prediction = pltData['plt_past_predict'][np.isin(pltData['plt_past_predict_index'], range(minI, maxI))]
        rmse = math.sqrt(mean_squared_error(ground_truth, prediction))
    print('Train Score: %.2f RMSE' % (rmse))
    pltData['pltErrors'] = np.append(pltData['pltErrors'], rmse)

    if len(pltData['pltErrors']) > len(pltData['pltErrorIndex']):
        pltData['pltErrors'] = np.delete(pltData['pltErrors'], 0)

    print(pltData['pltErrorIndex'] )
    #-----------


    return pltData


run_data = ml_data_x

def lstm_predict(ix, input_data, ahead):

    n_ts_features = len(ts_features) + 1  # features: value, week, month, etc
    multi_steps_predictions = np.array([])

    timesteps = int(input_data.shape[1] / n_ts_features)
    # reshape input pattern to [samples, timesteps, features]
    step_input = input_data.reshape(1, timesteps,  n_ts_features)

    while multi_steps_predictions.shape[0] < ahead:

        step_predict = model_lstm.predict(step_input)
        step_predict = step_predict.flatten()
        # print(step, step_predict)
        multi_steps_predictions = np.append(multi_steps_predictions, step_predict)

        # build next step
        n_out = step_predict.shape[0]
        features = run_data[(ix+1):(ix+n_out+1), -(n_ts_features - 1):]
        step_predict = step_predict.reshape((n_out, 1))
        step_predict = np.concatenate((step_predict, features), axis=1)

        step_input = np.reshape(step_input, (step_input.shape[1], n_ts_features))

        step_input = np.concatenate((step_input, step_predict))
        step_input = step_input[n_out:, :]
        step_input = step_input.reshape(1, timesteps, n_ts_features)
    model_lstm.reset_states()

    multi_steps_predictions = multi_steps_predictions[0:plt_step_ahead]
    # print('multi_steps_predictions:', multi_steps_predictions) # multi_steps_predictions.shape = (1, x)
    return multi_steps_predictions


def plot_figure(pltData):
    grid = plt.GridSpec(3, 1)
    plt.subplot(grid[0:2, 0])

    plt.plot(pltData['plt_value_index'], pltData['plt_value'], marker='.', c="grey", alpha=0.7)
    plt.plot(pltData['plt_ahead_predict_index'], pltData['plt_ahead_predict'], marker='.', c="r", linestyle='', alpha=0.4)
    plt.plot(pltData['plt_past_predict_index'][0:(len(pltData['plt_past_predict_index']) - 1)],
             pltData['plt_past_predict'][0:(len(pltData['plt_past_predict_index']) - 1)], marker='.', c="y", alpha=0.4)
    plt.title('plt_step_ahead:' + str(plt_step_ahead) + ', plt_max_back:' + str(plt_max_back), fontsize=10)
    plt.suptitle('Forecasting metric', fontsize=12)

    # plt.axhline(y=plotCrossThreshold, color='r', alpha=0.2)
    # plt.plot(pltData['plt_cross_index'], pltData['plt_cross'], marker='^', linestyle="", alpha=0.05)
    # plt.plot(pltData['pltWarnIndex'], pltData['pltWarns'], marker='x', linestyle="", color='r', alpha=0.5)

    plt.subplot(grid[2, 0])
    plt.plot(pltData['pltErrorIndex'], pltData['pltErrors'], marker='.', c="orange", alpha=0.2)

    plt.show()
    plt.pause(plt_pause)  # Note this correction
    # mplt.clf()



def run_lstm():

    plt_data = new_plt_data()

    run_data = ml_data_x

    n_ts_features = len(ts_features) + 1

    for ix in range(0, len(run_data)):

        # ix = 0
        input_data = run_data[[ix]]

        multi_steps_predictions = lstm_predict(ix, input_data, plt_step_ahead)
        multi_steps_predictions = ts_df_scaler.inverse_transform([multi_steps_predictions])
        multi_steps_predictions = multi_steps_predictions.flatten()

        actual_value = input_data[0][-n_ts_features] # last element of input
        actual_value = np.reshape(actual_value, (1, 1))
        actual_value = ts_df_scaler.inverse_transform(actual_value)

        print(actual_value)

        plt_data = get_plt_data(ix, actual_value, multi_steps_predictions, plt_data)

        plot_figure(plt_data)


run_lstm()



# --------------------------------------------------------------------------

def run_ets():

    import k_r.k_rpy2 as ets_py
    reload(ets_py)

    pltData = new_plt_data()

    n_ts_features = len(ts_features) + 1


    for ix in range(0, len(run_data)):

        # ix = 0
        if ix==0:
            input_data = run_data[:1, (run_data.shape[1] - n_ts_features)]
        else:
            # ix = 1
            input_data = run_data[:ix, (run_data.shape[1] - n_ts_features)]


        multi_steps_predictions = ets_py.ets_predict(input_data, ts_frequency, plt_step_ahead)
        print(multi_steps_predictions)
        multi_steps_predictions = ts_df_scaler.inverse_transform([multi_steps_predictions]).flatten()

        # print(multi_steps_predictions)

        actual_value = input_data[-1] # last element of input
        actual_value = np.reshape(actual_value, (1, 1))
        actual_value = ts_df_scaler.inverse_transform(actual_value)

        pltData = get_plt_data(ix, actual_value, multi_steps_predictions, pltData)

        if ix > 10:

            plot_figure(pltData)

# ---------------------------------------------------------------

import time

def run_prophet():

    plt_data = new_plt_data()

    run_data = ts_df
    run_data['ds'] = ts_df.index
    run_data['y'] = ts_df.iloc[:, 1]
    my_model = None

    for ix in range(1000, len(run_data)):

        # ix = 100

        print('ix', ix)

        t = time.clock()

        if my_model is None:
            input_data = run_data.iloc[:ix, ]
            my_model = prophet.build_model(input_data)
        elif ix % 100 == 0:
            input_data = run_data.iloc[:ix, ]
            my_model = prophet.build_model(input_data)


        current_date = run_data.index.date[ix]
        multi_steps_predictions = prophet.predict(my_model, ts_sample_frequency, current_date, plt_step_ahead)
        print(multi_steps_predictions)
        multi_steps_predictions = ts_df_scaler.inverse_transform([multi_steps_predictions]).flatten()

        elapsed_time = time.clock() - t
        print(elapsed_time)

        # print(multi_steps_predictions)

        actual_value = run_data.iloc[ix, -1]
        actual_value = ts_df_scaler.inverse_transform(actual_value)

        plt_data = get_plt_data(ix, actual_value, multi_steps_predictions, plt_data)

        if ix > 10:
            plot_figure(plt_data)



# run_prophet()