
# conda install -c conda-forge fbprophet

# https://facebook.github.io/prophet/docs/installation.html
# use conda install, the pip install require 'Microsoft Visual C++ Build Tools' on windows

# -----------------------------------------------

import k_lstm.my_utils as my_utils
ts_df = my_utils.get_object('k_lstm/data_temp/ts_df.pkl')


import pandas as pd
from fbprophet import Prophet


def build_model(y_df):

    # set the uncertainty interval to 95% (the Prophet default is 80%)
    # model = Prophet() #instantiate Prophet
    my_model = Prophet(interval_width=0.95)

    return my_model.fit(y_df) #fit the model with your dataframe


def predict(y_df, ts_sample_frequency, head):

    my_model = build_model(y_df)

    future_dates = my_model.make_future_dataframe(periods=head, freq=ts_sample_frequency)
    # future_dates.tail()

    forecast_output = my_model.predict(future_dates)

    forecast_output = forecast_output[len(y_df):]


    if False:
        forecast_output[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        my_model.plot(forecast_output, uncertainty=True)

    return forecast_output['yhat']









