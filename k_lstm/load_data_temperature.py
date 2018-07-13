import os as os

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime



# --------- parse datetime-------------------

def parse_date(date):
    if date == "null" or date == "":
        return None
    try:
        # '2012-10-01 12:00:00'
        dt = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        return dt
    except ValueError as e:
        try:
            dt = pd.to_datetime(int(date), unit='ns')
            return dt
        except ValueError as e:
            print("bad : ", date, e)
            return None



def load_data(data_path):


    #--------- load data --------------

    # Historical Hourly Weather Data 2012-2017
    # https://www.kaggle.com/selfishgene/historical-hourly-weather-data

    files = os.listdir(data_path)
    df = pd.DataFrame()
    for f in files:
        print("loading data: ", f)
        temp = pd.read_csv(data_path + f, header=0, sep=',')
        df = df.append(temp, ignore_index=True)


    print(df.describe())
    df.columns


    df['timestamp'] = df['datetime'].apply(lambda x: parse_date(x))
    df = df.sort_values('timestamp')

    df.head(5)

    return df


def get_ts(data_path):

    df = load_data(data_path)

    # ------------ get input data ----------------

    ts_df = df.iloc[:, [df.columns.get_loc('timestamp'), 2]] # Portland temperature

    #ts_df = ts_df[1:20001]
    ts_sample_frequency = '60min' # original
    ts_sample_frequency = 'D'


    # -------------plot input data ------------

    plt.plot(ts_df.iloc[:, 0], ts_df.iloc[:, 1].values)
    print(ts_df.shape)

    return ts_df, ts_sample_frequency

