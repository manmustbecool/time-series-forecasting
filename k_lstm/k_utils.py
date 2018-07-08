
import os as os
from keras.models import model_from_yaml


def save_model(model, dir=''):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(dir + "model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5 (pip install h5py)
    model.save_weights(dir + "model.weight")
    print("in cwd", os.getcwd())
    print("Saved model to disk: ", dir + "model.yaml")


def load_model(dir=''):
    print("in cwd", os.getcwd())
    # load YAML and create model
    yaml_file = open(dir + 'model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(dir + "model.weight")
    print("Loaded model from disk: ", dir + 'model.yaml')
    return loaded_model
