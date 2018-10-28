import numpy as np

def fast_build_poly(x, poly_degree_minus_one, degree):
    """Fast polynomial basis functions for input data x to be used in loops only."""
    return np.c_[poly_degree_minus_one, np.power(x, degree)]

def split_data(x, y, ratio, seed=23):
    """Split the given datasets based on the split ratio."""
    np.random.seed(seed)
    N = y.shape[0]
    indices = np.random.permutation(N)
    index_split = int(np.floor(ratio * N))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def build_k_indices(y, k_fold, seed=23):
    """Build k indices for k-fold cross validation."""
    N = y.shape[0]
    interval = int(N / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
