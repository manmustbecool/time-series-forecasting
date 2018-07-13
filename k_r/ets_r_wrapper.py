import subprocess
import os


from importlib import reload

import k_config as k_config
reload(k_config)

# Define command and arguments
command = k_config.r_script

path2script = os.getcwd()+'\ets.R'
print('path2script', path2script)


def ets_r_prediction(ahead, ts_v, ts_frequency=1):
    # Build subprocess command
    # ['<command_to_run>', '<path_to_script>', 'arg1' , 'arg2', 'arg3', 'arg4']
    # Variable number of args in a list
    # each arg must be a string
    args = [str(ahead), str(ts_frequency)] + list(map(str, ts_v))
    cmd = [command, path2script] + args

    # check_output will run the command and store to result
    output = subprocess.check_output(cmd, universal_newlines=True)

    for line in output.splitlines():
        print(line)


if True:
    v = [27, 27, 7, 24, 39, 40, 24, 45, 36, 37, 31, 47, 16, 24, 6, 21, 35, 36, 21, 40, 32, 33, 27, 42, 14, 21, 5, 19, 31, 32, 19, 36, 29, 29, 24, 42, 15]
    ets_r_prediction(5, v, 12)








