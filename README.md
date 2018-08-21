# Time series forecasting

### Time series forecasting

  * lstm (Long Short Term Memory) 

  * ets (Exponential smoothing state space)

  * stl (Seasonal and Trend decomposition using Loess)

  * prophet


### 

![Figure_1.png](images/Figure_1.png)

 	
~~~~

ts_df: dataframe['timestamp', 'kpivalue']

ts_df_scaler: scaler of kpivalue

ts_features: ['hourly', 'weekly', ...]

~~~~


### project

set 'lstm-time-series' as working directory

  * In Pycharm

in pycharm > file > setting > console > working directory )

  * In python script

`
import os

os.chdir("C:\\...\\lstm-time-series")

print(os.getcwd())
`
