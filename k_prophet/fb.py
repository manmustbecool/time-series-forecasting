
# conda install -c conda-forge fbprophet

# https://facebook.github.io/prophet/docs/installation.html
# use conda install, the pip install require 'Microsoft Visual C++ Build Tools' on windows

# -----------------------------------------------

import k_lstm.my_utils as my_utils
ts_df= my_utils.get_object('k_lstm/data_temp/ts_df.pkl')


ts_df['ds'] = ts_df.index
ts_df['y'] = ts_df.iloc[:,1]


import pandas as pd
from fbprophet import Prophet


# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.95)




# model = Prophet() #instantiate Prophet
my_model.fit(ts_df) #fit the model with your dataframe


future_dates = my_model.make_future_dataframe(periods=50, freq='D')
future_dates.tail()


forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



my_model.plot(forecast,
              uncertainty=True)

my_model.plot_components(forecast)




