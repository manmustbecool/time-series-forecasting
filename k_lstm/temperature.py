import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data_path = '../input/'

print(os.listdir(data_path))

files = os.listdir(data_path)
df = pd.DataFrame()
for f in files: # [2:3] only load temperature.csv for the time being
    # do something
    print("loading data: ", f)
    temp = pd.read_csv(data_path + f, header=0, sep=',')
    df = df.append(temp, ignore_index=True)

df.describe()
df.columns

# --------- parse datetime-------------------
from datetime import datetime
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


df['timestamp'] = df['datetime'].apply(lambda x: parse_date(x))
df = df.sort_values('timestamp')

df.head(5)

yDf = df[0:501]
yDf = yDf.drop(yDf.index[0:1])
y = yDf.iloc[:,2]
plt.plot(y)


ts_data = df.iloc[:, [df.columns.get_loc('timestamp'), 2]]

ts_df = ts_data[0:5000]
ts_df_frequency = '60min'


print(ts_data.shape)