import pickle


def get_var_name(var):
    """
    return python object by var name
    """
    for k, v in globals().items():
        if v is var:
            print("found name:", k)
            return k
    print("did not find var name")


def save_object(obj, filename=None):
    """
    save Python object to .pkl file
    """
    filename = filename + '.pkl'
    if filename is None:
        filename = get_var_name(obj)+'.pkl'
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print('Object', filename, "is saved")


def get_object(filename):
    """
    return Python object by reading .pkl file
    """
    with open(filename, 'rb') as input:
        return pickle.load(input)
