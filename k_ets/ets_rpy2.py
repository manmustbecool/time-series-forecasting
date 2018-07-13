# ----------- install pry2 ----------------------

# set R_HOME=C:\Program Files\R\R-3.4.1\bin\
# ECHO %R_HOME%
# set PATH=%R_HOME%;%PATH%
# ECHO %PATH%

# pip install rpy2-2.9.4-cp36-cp36m-win_amd64.whl

# pip install tzlocal # is also required

#------------- load R forecast package -----------------------------------------

import os
os.environ['R_USER'] = 'C:\\Program Files\\R\R-3.5.1\\bin'


# type '.libPaths()' in R to find all lib paths
from rpy2.robjects.packages import importr
base = importr('base')
lib_paths = base._libPaths()
print(lib_paths)


from rpy2.robjects.packages import importr
utils = importr("utils")
d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
try:
    r_forecast = importr('forecast', robject_translations = d, lib_loc = "C:/Users/my/Documents/R/win-library/3.5")
    print('xx')
except:
    r_forecast = importr('forecast', robject_translations = d, lib_loc = "C:/Program Files/R/R-3.5.1/library" )
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
    print(ro_ts)

    ro_fit = r_forecast.ets(ro_ts)

    forecast_output = r_forecast.forecast(ro_fit, h=head)

    output = np.array(forecast_output.rx('mean')).flatten()

    print(output)

    return output


if False:

    y = [27, 27, 7, 24, 39, 40, 24, 45, 36, 37, 31, 47, 16, 24, 6, 21, 35, 36, 21, 40, 32, 33, 27, 42, 14, 21, 5, 19, 31, 32, 19, 36, 29, 29, 24, 42, 15]
    frequency = 3
    head =4

    y = np.array(y)

    print(ets_predict(y, frequency, 3))



    y = ts_df.iloc[:, 1].values



