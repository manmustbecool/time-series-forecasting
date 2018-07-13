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
"""


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # pandas shift() function can be used to create copies of columns that are pushed forward (rows of  NaN values added to the front)
    # or pulled back (rows of NaN values added to the end).

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
    print(' ')

    return agg


# transform series into train and test sets for supervised learning
def get_ml_data(ml_data, n_in, n_out):

    supervised = series_to_supervised(ml_data, n_in, n_out)
    supervised_values = supervised.values

    # split into train and test sets
    n_vars = 1 if type(ml_data) is list else ml_data.shape[1]

    ml_data_x = supervised_values[:, 0:(n_in * n_vars)]
    print('ml_data_x shape:', ml_data_x.shape)

    ml_data_y = supervised_values[:, (n_in * n_vars):]

    # keep only value columns (i.e var1) for _y data.
    value_column_ix = np.arange(0, (ml_data_y.shape[1]), n_vars)
    ml_data_y = ml_data_y[:, value_column_ix]

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

    data = data.iloc[:,:-4]
    data.shape
    # expected input data shape: (batch_size / samples, timesteps, data_dim / features)
    data = np.array(data)
    np.reshape(data, (6, 3, 2))

    valueX, valueY = prepare_data(values, 3, 2)
    print(valueX)
    print(valueY)

#----------------------------------------------------

"""

"""
def get_train_test_data(ml_data_x, ml_data_y, train_size_rate):

    # split into train and test sets
    trainSize = int(len(ml_data_x) * train_size_rate)

    trainX, testX = ml_data_x[0:trainSize, :], ml_data_x[trainSize:len(ml_data_x), :]
    trainY, testY = ml_data_y[0:trainSize, :], ml_data_y[trainSize:len(ml_data_y), :]

    print('train data:', trainX.shape, trainY.shape)
    print('test data:', testX.shape, testY.shape)

    return trainX, trainY, testX, testY


#----------------

def add_time_based_feature(ml_data, time_features=[]):

    for feature in time_features:
        if feature == 'hour':
            ml_data['hour'] = ml_data.index.hour + 1
        if feature == 'weekday':
            ml_data['weekday'] = ml_data.index.weekday + 1  # + 1 to avoid 0
        if feature == 'month':
            ml_data['month'] = ml_data.index.month + 1  # + 1 to avoid 0

    return ml_data


from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from k_lstm.my_utils import save_object


def prepare_data(ts_df, ts_sample_frequency, temp_data_folder, look_back, look_forward, train_size_rate, ts_features=[]):

    # look_back = 10 ; look_forward = 5; train_size_rate = 0.7

    # ts_df is a dataframe
    ts_df = ts_df.set_index(keys='timestamp')
    print('ts_df row count:', len(ts_df))
    ts_df.dropna(inplace=True)
    print('Pre-filter ts_df row count:', len(ts_df))

    if False:
        ts = ts_df.iloc[:, 0]

        ts.plot()
        ts.plot(marker='.', linestyle="")
        plt.plot(range(0, len(ts)), ts.values)

    # normalize the dataset
    # scale
    ts_df = ts_df.astype('float')
    ts_df_scaler = MinMaxScaler(feature_range=(0, 1))
    ts_df['scaled_v'] = ts_df_scaler.fit_transform(ts_df)

    # fix missing ts_df
    ts_df = ts_df.resample(ts_sample_frequency).mean()
    print('Fixed missing data, ts_df row count:', len(ts_df))

    plt.plot(ts_df.iloc[:, 1])

    ml_data = ts_df.loc[:, ['scaled_v']] # ml_data is a dataframe

    ml_data = add_time_based_feature(ml_data, ts_features)

    ml_data_x, ml_data_y = get_ml_data(ml_data, look_back, look_forward) # X and Y are numpy.ndarray

    train_x, train_y, test_x, test_y = get_train_test_data(ml_data_x, ml_data_y, train_size_rate)

    # -------- save data in temp data folder -----------
    save_object(train_x, temp_data_folder+"train_x")
    save_object(train_y, temp_data_folder+"train_y")
    save_object(test_x, temp_data_folder + "test_x")
    save_object(test_y, temp_data_folder + "test_y")
    save_object(ml_data_x, temp_data_folder+"ml_data_x")
    save_object(ts_df, temp_data_folder + "ts_df")
    save_object(ts_df_scaler, temp_data_folder+"ts_df_scaler")
    save_object(ts_features, temp_data_folder + "ts_features")




