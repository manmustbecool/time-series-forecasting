import numpy as np

import math
from sklearn.metrics import mean_squared_error

from importlib import reload

import k_prophet.prophet as prophet
reload(prophet)

plotStepAhead = 60
plotMaxBack = 2000
plt_pause = 0.1

plotCrossStepAhead = 5
plotCrossThreshold = 300
plotWarnThreshold = 10

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


def new_pltData():
    x = dict(pltValueIndex=np.array([]), pltValues=np.array([]), pltPastPredicts=np.array([]), pltPastPredictIndex=np.array([]),
               pltAheadPredicts=np.array([]), pltAheadPredictIndex=np.array([]), pltCrossIndex=np.array([]), pltCrosses=np.array([]),
               pltWarns=np.array([]), pltWarnIndex=np.array([]), pltErrorIndex=np.array([]), pltErrors=np.array([]))
    return x

plt.ion() ## Note this correction

def get_pltData(ix, actual_value, multi_steps_predictions, pltData):

    # ix = 0;

    print("ix:", ix)

    # ---- pltValues -------
    pltData['pltValues'] = np.append(pltData['pltValues'], actual_value)
    if len(pltData['pltValues']) > plotMaxBack:
        pltData['pltValues'] = np.delete(pltData['pltValues'], 0)

    pltData['pltValueIndex'] = np.arange((ix + 1 - len(pltData['pltValues'])), (ix + 1))

    # print(pltData['pltValues'])
    # print(pltData['pltValueIndex'])

    # ---- pltPastPredicts -------
    temp = pltData['pltPastPredicts']
    temp = np.append(temp, multi_steps_predictions[(plotCrossStepAhead-1)])
    if len(temp) > (plotMaxBack + plotCrossStepAhead):
        temp = np.delete(temp, 0)
    pltData['pltPastPredicts'] = temp

    pltData['pltPastPredictIndex'] = np.arange((ix + plotCrossStepAhead + 1 - len(temp)), (ix + plotCrossStepAhead + 1))
    # print('plotPredicts:', pltPastPredicts)
    # print(pltPastPredictIndex)

    # ---- pltAheadPredicts -------

    pltData['pltAheadPredicts'] = multi_steps_predictions.flatten()
    pltData['pltAheadPredictIndex'] = np.arange((ix + 1), (ix + 1 + plotStepAhead))
   # else:
   #     pltData['pltAheadPredicts'] = np.append(np.array(pltData['pltPastPredicts'][-2]), multiStepPredicts.flatten())
   #     pltData['pltAheadPredictIndex'] = np.arange((ix + 1 - 1), (ix + 1 + plotStepAhead))

    pltData['pltAheadPredictIndex'] = np.array(pltData['pltAheadPredictIndex'])

    # ---- pltCrosses -------
    crossIndex = (pltData['pltAheadPredicts'] > plotCrossThreshold)
    crossIndex = np.append(np.full(plotCrossStepAhead, False), crossIndex[plotCrossStepAhead:])

    crosses = pltData['pltAheadPredicts'][crossIndex]
    crossIndex = pltData['pltAheadPredictIndex'][crossIndex]

    pltData['pltCrosses'] = np.append(pltData['pltCrosses'], crosses)
    pltData['pltCrossIndex'] = np.append(pltData['pltCrossIndex'], crossIndex)

    # clean old cross data
    pltData['pltCrosses'] = pltData['pltCrosses'][pltData['pltCrossIndex'] >= pltData['pltPastPredictIndex'][0]]
    pltData['pltCrossIndex'] = pltData['pltCrossIndex'][pltData['pltCrossIndex'] >= pltData['pltPastPredictIndex'][0]]

    # calculate warns
    crossIndexUnique = np.asarray(np.unique(pltData['pltCrossIndex'], return_counts=True)).T
    crossIndexUnique = crossIndexUnique[crossIndexUnique[:, 1] > plotWarnThreshold]
    pltData['pltWarns'] = np.full(len(crossIndexUnique), plotCrossThreshold)
    pltData['pltWarnIndex'] = crossIndexUnique[:, 0]

    # ----  calculate  error ----------------
    # minI = max(min(pltData['pltPastPredictIndex']), min(pltData['pltValueIndex']))
    # maxI = min(max(pltData['pltPastPredictIndex']), max(pltData['pltValueIndex']))+1
    # pltData['pltErrorIndex'] = np.arange(minI, maxI)
    # rmse = 0
    # if len(pltData['pltValues']) > 2:
    #     groundTruth= pltData['pltValues'][np.isin(pltData['pltValueIndex'], range(minI,maxI))]
    #     prediction= pltData['pltPastPredicts'][np.isin(pltData['pltPastPredictIndex'], range(minI, maxI))]
    #     rmse = math.sqrt(mean_squared_error(groundTruth, prediction))
    # print('Train Score: %.2f RMSE' % (rmse))
    # pltData['pltErrors'] = np.append(pltData['pltErrors'], rmse)
    #
    # if len(pltData['pltErrors']) > len(pltData['pltErrorIndex']):
    #     pltData['pltErrors'] = np.delete(pltData['pltErrors'], 0)

    # print(pltData['pltErrorIndex'] )
    #-----------


    return pltData


run_data = ml_data_x

def lstm_predict(ix, input_data, ahead):

    n_ts_features = len(ts_features) + 1
    multi_steps_predictions = np.array([])

    timesteps = int(input_data.shape[1] / n_ts_features)
    # reshape input pattern to [samples, timesteps, features]
    step_input = input_data.reshape(1, timesteps,  n_ts_features)

    while multi_steps_predictions.shape[0] < ahead:

        step_predict = model_lstm.predict(step_input)
        step_predict = step_predict.flatten()
        n_out = step_predict.shape[0]
        # print(step, step_predict)
        multi_steps_predictions = np.append(multi_steps_predictions, step_predict)

        # build next step
        features = run_data[(ix+1):(ix + n_out+1), -(n_ts_features - 1):]
        step_predict = step_predict.reshape((n_out, 1))
        step_predict = np.concatenate((step_predict, features), axis=1)

        step_input = np.reshape(step_input, (step_input.shape[1], n_ts_features))

        step_input = np.concatenate((step_input, step_predict))
        step_input = step_input[n_out:,:]
        step_input = step_input.reshape(1, timesteps, n_ts_features)
    model_lstm.reset_states()

    multi_steps_predictions = multi_steps_predictions[0:plotStepAhead]
    # print('multi_steps_predictions:', multi_steps_predictions) # multi_steps_predictions.shape = (1, x)
    return multi_steps_predictions


def plot_figure(pltData):
    grid = plt.GridSpec(3, 1)
    plt.subplot(grid[0:2, 0])

    plt.plot(pltData['pltValueIndex'], pltData['pltValues'], marker='.', c="black", alpha=0.5)
    plt.plot(pltData['pltAheadPredictIndex'], pltData['pltAheadPredicts'], marker='.', c="r", linestyle='', alpha=0.3)
    plt.plot(pltData['pltPastPredictIndex'][0:(len(pltData['pltPastPredictIndex']) - 1)],
             pltData['pltPastPredicts'][0:(len(pltData['pltPastPredictIndex']) - 1)], marker='.', c="y", alpha=0.7)

    # plt.axhline(y=plotCrossThreshold, color='r', alpha=0.2)
    # plt.plot(pltData['pltCrossIndex'], pltData['pltCrosses'], marker='^', linestyle="", alpha=0.05)
    # plt.plot(pltData['pltWarnIndex'], pltData['pltWarns'], marker='x', linestyle="", color='r', alpha=0.5)

    plt.subplot(grid[2, 0])
    # plt.plot(pltData['pltErrorIndex'], pltData['pltErrors'], marker='.', c="orange", alpha=0.5)

    plt.show()
    plt.pause(plt_pause)  # Note this correction
    # mplt.clf()




def run_lstm():

    pltData = new_pltData()
    run_data = ml_data_x
    n_ts_features = len(ts_features) + 1

    for ix in range(0, len(run_data)):

        # ix = 0
        input_data = run_data[[ix]]

        multi_steps_predictions = lstm_predict(ix, input_data, plotStepAhead)
        multi_steps_predictions = ts_df_scaler.inverse_transform([multi_steps_predictions])
        multi_steps_predictions = multi_steps_predictions.flatten()

        actual_value = input_data[0][-n_ts_features] # last element of input
        actual_value = np.reshape(actual_value, (1, 1))
        actual_value = ts_df_scaler.inverse_transform(actual_value)

        print(actual_value)

        pltData = get_pltData(ix, actual_value, multi_steps_predictions, pltData)

        plot_figure(pltData)


run_lstm()


def run_ets():

    import k_r.k_rpy2 as ets_py
    reload(ets_py)

    pltData = new_pltData()

    n_ts_features = len(ts_features) + 1


    for ix in range(0, len(run_data)):

        # ix = 0
        if ix==0:
            input_data = run_data[:1, (run_data.shape[1] - n_ts_features)]
        else:
            # ix = 1
            input_data = run_data[:ix, (run_data.shape[1] - n_ts_features)]


        multi_steps_predictions = ets_py.ets_predict(input_data, ts_frequency, plotStepAhead)
        print(multi_steps_predictions)
        multi_steps_predictions = ts_df_scaler.inverse_transform([multi_steps_predictions]).flatten()

        # print(multi_steps_predictions)

        actual_value = input_data[-1] # last element of input
        actual_value = np.reshape(actual_value, (1, 1))
        actual_value = ts_df_scaler.inverse_transform(actual_value)

        pltData = get_pltData(ix, actual_value, multi_steps_predictions, pltData)

        if ix > 10:

            plot_figure(pltData)

# ---------------------------------------------------------------

import time

def run_prophet():

    pltData = new_pltData()

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
        multi_steps_predictions = prophet.predict(my_model, ts_sample_frequency, current_date, plotStepAhead)
        print(multi_steps_predictions)
        multi_steps_predictions = ts_df_scaler.inverse_transform([multi_steps_predictions]).flatten()

        elapsed_time = time.clock() - t
        print(elapsed_time)

        # print(multi_steps_predictions)

        actual_value = run_data.iloc[ix, -1]
        actual_value = ts_df_scaler.inverse_transform(actual_value)

        pltData = get_pltData(ix, actual_value, multi_steps_predictions, pltData)

        if ix > 10:
            plot_figure(pltData)



# run_prophet()