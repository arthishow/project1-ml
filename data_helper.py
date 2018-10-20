import numpy as np

def standardize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x-mean)/std
