# -*- coding: utf-8 -*-
"""implement a split_data function."""

import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    n_sample=int(np.floor(ratio*len(x)))
    x_copy=np.copy(x)
    y_copy=np.copy(y)
    np.random.seed(seed)
    np.random.shuffle(x_copy)
    x_train,x_test=np.split(x_copy, [n_sample])
    np.random.seed(seed)
    np.random.shuffle(y_copy)
    y_train,y_test=np.split(y_copy, [n_sample])
    return x_train,x_test,y_train,y_test

