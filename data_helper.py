import numpy as np

def remove_missing_values(x):
    """Remove columns contaniing outliers (-999)."""
    x_clean = x.copy()
    x_clean = x_clean[:, [np.count_nonzero(x.T[i] == -999) == 0 for i in range (x.shape[1])]]
    x_clean = standardize(x_clean)
    return x_clean

def set_missing_values_to_mean(x):
    """Set outliers values (-999) to the mean of their respective column."""
    x_clean = x.copy()
    x_clean[x_clean == -999] = np.nan
    col_means = np.nanmean(x_clean, axis=0)
    inds = np.where(np.isnan(x_clean))
    x_clean[inds] = np.take(col_means, inds[1])
    x_clean = standardize(x_clean)
    return x_clean

def standardize(x):
    """Normalize the given dataset."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x-mean)/std

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

def build_poly(x, degree):
    """Polynomial basis functions for input data x."""
    poly = np.ones((len(x), 1))
    for d in (np.arange(degree) + 1):
        poly = np.c_[poly, np.power(x, d)]
    return poly
