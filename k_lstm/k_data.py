import numpy as np
import pandas as pd

# --------------------------------- prepare time series data ---------------------------

"""
Frame a time series as a supervised learning dataset.
Arguments:
	data: Sequence of observations as a list or NumPy array.
	n_in: Number of lag observations as input (X).
	n_out: Number of observations as output (y).
	dropnan: Boolean whether or not to drop rows with NaN values.
Returns:
	Pandas DataFrame of series framed for supervised learning.

pandas shift() function can be used to create copies of columns that are pushed forward (rows of 
NaN values added to the front) or pulled back (rows of NaN values added to the end).
"""


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    row_count = len(agg)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    print(agg.head(2))
    print('Number of features/vars: ', n_vars)
    if len(agg) == 0:
        print(" !! ML data is empty !!")
    print('series_to_supervised. Row count:', row_count, "After drop row containing NaN values:", len(agg),
          '-- Drop rate:', str(1-(len(agg)/row_count)) )
    print('-------------------------- ')
    return agg


# transform series into train and test sets for supervised learning
def prepare_data(mlData, n_in, n_out):
    supervised = series_to_supervised(mlData, n_in, n_out)
    supervised_values = supervised.values

    # split into train and test sets
    n_vars = 1 if type(mlData) is list else mlData.shape[1]

    ml_data_x = supervised_values[:, 0:(n_in * n_vars)]
    print('ml_data_x shape:', ml_data_x.shape)

    ml_data_y = supervised_values[:, (n_in * n_vars):]
    print('ml_data_y shape:', ml_data_y.shape)
    return ml_data_x, ml_data_y


if False:
    values = [x for x in range(10)]
    data = series_to_supervised(values, 2, 1)
    print(data)

    valueX, valueY = prepare_data(values, 2, 1)
    print(valueX)
    print(valueY)

if False:
    raw = pd.DataFrame()
    raw['ob1'] = [x for x in range(10)]
    raw['ob2'] = [x for x in range(50, 60)]
    values = raw.values
    data = series_to_supervised(values, 3, 2)
    print(data)

    valueX, valueY = prepare_data(values, 3, 2)
    print(valueX)
    print(valueY)


#----------------------------------------------------

"""

"""
def get_train_test_data(mlDataX, mlDataY):

    # split into train and test sets
    trainSize = int(len(mlDataX) * 0.70)

    trainX, testX = mlDataX[0:trainSize, :], mlDataX[trainSize:len(ml_data), :]
    trainY, testY = mlDataY[0:trainSize, :], mlDataY[trainSize:len(ml_data), :]

    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)

    return trainX, trainY, testX, testY


#----------------

from sklearn.preprocessing import MinMaxScaler
from importlib import reload

# import kaggle_kpi.temperature as data_source;
import kaggle_kpi.cell_kpi as data_source;
# reload(data_source)
ts_df = data_source.ts_df
ts_df_frequency = data_source.ts_df_frequency


# ts_df is a dataframe
ts_df = ts_df.set_index(keys='timestamp')
print('ts_df row count:', len(ts_df))
ts_df.dropna(inplace=True)
print('Pre-filter ts_df row count:', len(ts_df))

if False:
    ts = ts_df.iloc[:, 0]
    import matplotlib.pyplot as plt
    ts.plot()
    ts.plot(marker='.', linestyle="")
    plt.plot(range(0,len(ts)), ts.values)

# scale
ts_df = ts_df.astype('float')
# normalize the dataset
ts_df_scaler = MinMaxScaler(feature_range=(0, 1))
ts_df['scaled_v'] = ts_df_scaler.fit_transform(ts_df)


# fix missing ts_df
ts_df = ts_df.resample(ts_df_frequency).mean()
print('Fixed missing data, ts_df row count:', len(ts_df))

ml_data = ts_df.loc[:, ['scaled_v']] # mldata is a dataframe



look_back = 5
look_forward = 5

ml_data_x, ml_data_y = prepare_data(ml_data, look_back, look_forward) # X and Y are numpy.ndarray
trainX, trainY, testX, testY = get_train_test_data(ml_data_x, ml_data_y)



# -------- save data in temp data folder -----------

import kaggle_kpi.kaggle_config as kaggle_config
reload(kaggle_config)
temp_data_folder = kaggle_config.temp_data_folder

import kaggle_kpi.my_utils.my_utils as my_utils
reload(my_utils)

my_utils.save_object(trainX, temp_data_folder+"train_x")
my_utils.save_object(trainY, temp_data_folder+"train_y")
my_utils.save_object(testX, temp_data_folder+"testX")
my_utils.save_object(ml_data_x, temp_data_folder+"ml_data_x")
my_utils.save_object(ts_df_scaler, temp_data_folder+"ts_df_scaler")




