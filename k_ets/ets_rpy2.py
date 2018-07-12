# ----------- install pry2 ----------------------

# set R_HOME=C:\Program Files\R\R-3.4.1\bin\
# ECHO %R_HOME%
# set PATH=%R_HOME%;%PATH%
# ECHO %PATH%

# pip install rpy2-2....

#------------- load R forecast package -----------------------------------------

import os
os.environ['R_USER'] = 'C:\\Program Files\\R\R-3.4.1\\bin'


# type '._libPaths()' in R to find all lib paths
from rpy2.robjects.packages import importr
base = importr('base')
lib_paths = base._libPaths()
print(lib_paths)


from rpy2.robjects.packages import importr
utils = importr("utils")
d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
try:
    r_forecast = importr('forecast', robject_translations = d, lib_loc = "C:/Users/emiewag/Documents/R/win-library/3.4")
    print('xx')
except:
    r_forecast = importr('forecast', robject_translations = d, lib_loc = "C:/Program Files/R/R-3.4.1/library" )
    print('yy')


# ---------------------  ets ------------------------------


from rpy2.robjects import pandas2ri
pandas2ri.activate()

import rpy2.robjects as robjects
import numpy as np

def ets_predict(y, ts_frequency, head):

    # convert y to R ts object
    ro_ts = robjects.r('ts')
    ro_ts = ro_ts(y, frequency=ts_frequency)

    ro_fit = r_forecast.ets(ro_ts)
    forecast_output = r_forecast.forecast(ro_fit, h=head)

    print(forecast_output.rx('mean'))

    return np.array(forecast_output.rx('mean')).flatten()


if False:

    y = [0, 1, 1, 1, 2, 3, 4, 5, 6]
    frequency = 3
    head =5

    y = np.array(y)

    print(ets_predict(y, frequency, 5))



    y = ts_df.iloc[:, 1].values



