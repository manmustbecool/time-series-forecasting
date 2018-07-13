# ----------- install pry2 ----------------------

# set R_HOME=C:\Program Files\R\R-3.4.1\bin\
# ECHO %R_HOME%
# set PATH=%R_HOME%;%PATH%
# ECHO %PATH%

# pip install rpy2-2.9.4-cp36-cp36m-win_amd64.whl

# pip install tzlocal # is also required

#------------- load R forecast package -----------------------------------------


from importlib import reload

import k_config as k_config
reload(k_config)


import os
os.environ['R_USER'] = k_config.r_user


# type '.libPaths()' in R to find all lib paths
from rpy2.robjects.packages import importr
base = importr('base')
lib_paths = base._libPaths()
print(lib_paths)


from rpy2.robjects.packages import importr
utils = importr("utils")
d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
try:
    r_forecast = importr('forecast', robject_translations = d, lib_loc = k_config.r_lib_loc_1)
    r_stats = importr('stats', robject_translations=d, lib_loc=k_config.r_lib_loc_1)
    print('load r lib in', k_config.r_lib_loc_1)
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
    # print(ro_ts)

    ro_fit = r_forecast.ets(ro_ts)

    forecast_output = r_forecast.forecast(ro_fit, h=head)

    output = np.array(forecast_output.rx('mean')).flatten()

    return output

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def stl_predict(y, ts_frequency, head):

    if y.shape[0] < 5:
        return np.full(head, mean(y))

    if y.shape[0] < (ts_frequency*2+2):
        ts_frequency = 2

    #rint(y.shape[0])

    # convert y to R ts object
    ro_ts = robjects.r('ts')
    ro_ts = ro_ts(y, frequency=ts_frequency)

    #print(ro_ts)

    ro_fit = r_stats.stl(ro_ts, "periodic", robust=True)

    forecast_output = r_forecast.forecast(ro_fit, h=head)

    output = np.array(forecast_output.rx('mean')).flatten()

    return output


if False:

    y = [27, 27, 7, 24, 39, 40, 24, 45, 36, 37, 31, 47, 16, 24, 6, 21, 35, 36, 21, 40, 32, 33, 27, 42, 14, 21, 5, 19, 31, 32, 19, 36, 29, 29, 24, 42, 15]
    frequency = 12
    head =5

    y = np.array(y)

    print(ets_predict(y, frequency, head))

    print(stl_predict(y, frequency, head))

    # [19.40121429  1.66500234 17.1365995  30.77005763 32.12580561]

    y = ts_df.iloc[:, 1].values



