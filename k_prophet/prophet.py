
# conda install -c conda-forge fbprophet

# https://facebook.github.io/prophet/docs/installation.html
# use conda install, the pip install require 'Microsoft Visual C++ Build Tools' on windows

# -----------------------------------------------

import k_lstm.my_utils as my_utils
ts_df = my_utils.get_object('k_lstm/data_temp/ts_df.pkl')

import numpy as np

import pandas as pd
from fbprophet import Prophet


def build_model(y_df):

    # set the uncertainty interval to 95% (the Prophet default is 80%)
    # model = Prophet() #instantiate Prophet
    my_model = Prophet(interval_width=0.95)

    my_model = my_model.fit(y_df)  # fit the model with your dataframe

    return my_model


import time

def predict(my_model, ts_sample_frequency, current_date, head):

    # head = 10

    future_dates = pd.date_range(start=current_date, periods=head + 1,  # An extra in case we include start
        freq=ts_sample_frequency)

    future_dates = pd.DataFrame(future_dates, index=np.arange(0, len(future_dates)))
    future_dates.columns = ['ds']

    future_dates = future_dates[1:]

    # future_dates = my_model.make_future_dataframe(periods=head, freq=ts_sample_frequency,  include_history=False)
    # future_dates.tail()

    forecast_output = my_model.predict(future_dates)


    if False:
        forecast_output[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        my_model.plot(forecast_output, uncertainty=True)

    return forecast_output['yhat']









