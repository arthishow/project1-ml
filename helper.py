import numpy as np

def preprocess_datasets(x_tr, y_tr, x_te, y_te):
    x_tr_0, y_tr_0, x_tr_1, y_tr_1, x_tr_2, y_tr_2, x_tr_3, y_tr_3 = split_dataset_by_jet_num(x_tr, y_tr)
    x_te_0, _, x_te_1, _, x_te_2, _, x_te_3, _ = split_dataset_by_jet_num(x_te, y_te)

    x_tr_0 = set_missing_values_to_mean(x_tr_0)
    x_tr_1 = set_missing_values_to_mean(x_tr_1)
    x_tr_2 = set_missing_values_to_mean(x_tr_2)
    x_tr_3 = set_missing_values_to_mean(x_tr_3)

    x_tr_0 = remove_correlated_columns(x_tr_0)
    x_tr_1 = remove_correlated_columns(x_tr_1)
    x_tr_2 = remove_correlated_columns(x_tr_2)
    x_tr_3 = remove_correlated_columns(x_tr_3)

    x_tr_0, mean_0, std_0 = standardize(x_tr_0)
    x_tr_1, mean_1, std_1 = standardize(x_tr_1)
    x_tr_2, mean_2, std_2 = standardize(x_tr_2)
    x_tr_3, mean_3, std_3 = standardize(x_tr_3)

    x_te_0 = (x_te_0-mean_0)/std_0
    x_te_1 = (x_te_1-mean_1)/std_1
    x_te_2 = (x_te_2-mean_2)/std_2
    x_te_3 = (x_te_3-mean_3)/std_3

    return x_tr_0, y_tr_0, x_tr_1, y_tr_1, x_tr_2, y_tr_2, x_tr_3, y_tr_3, x_te_0, x_te_1, x_te_2, x_te_3

def remove_correlated_columns(x, threshold=0.9):
    D = x.shape[1]
    column_index = []
    correlation_matrix = np.corrcoef(x, rowvar=False)
    for i in np.arange(D):
        for j in np.arange(i):
            if correlation_matrix[i, j] >= threshold:
                column_index.append(i)

    x = np.delete(x, column_index, 1)
    return x

def split_dataset_by_jet_num(x, y):
    x_0 = []
    y_0 = []
    unnecessary_columns_0 = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    x_1 = []
    y_1 = []
    unnecessary_columns_1 = [4, 5, 6, 12, 22, 26, 27, 28]
    x_2 = []
    y_2 = []
    unnecessary_columns_2 = [22]
    x_3 = []
    y_3 = []
    unnecessary_columns_3 = [22]
    for index, row in enumerate(x):
        if x[index, 22] == 0:
            x_0.append(np.delete(row, unnecessary_columns_0))
            y_0.append(y[index])
        elif x[index, 22] == 1:
            x_1.append(np.delete(row, unnecessary_columns_1))
            y_1.append(y[index])
        elif x[index, 22] == 2:
            x_2.append(np.delete(row, unnecessary_columns_2))
            y_2.append(y[index])
        else:
            x_3.append(np.delete(row, unnecessary_columns_3))
            y_3.append(y[index])

    x_0 = np.asarray(x_0)
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    x_3 = np.asarray(x_3)
    y_0 = np.asarray(y_0)
    y_1 = np.asarray(y_1)
    y_2 = np.asarray(y_2)
    y_3 = np.asarray(y_3)

    return x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3

def remove_missing_values(x):
    """Remove columns containing outliers (-999)."""
    x_clean = x.copy()
    x_clean = x_clean[:, [np.count_nonzero(x.T[i] == -999) == 0 for i in range (x.shape[1])]]
    return x_clean

#https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
def set_missing_values_to_mean(x):
    """Set outliers values (-999) to the mean of their respective column."""
    x_clean = x.copy()
    x_clean[x_clean == -999] = np.nan
    col_means = np.nanmean(x_clean, axis=0)
    inds = np.where(np.isnan(x_clean))
    x_clean[inds] = np.take(col_means, inds[1])
    return x_clean

def standardize(x):
    """Normalize the given dataset."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x-mean)/std, mean, std

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

def fast_build_poly(x, poly_degree_minus_one, degree):
    """Fast polynomial basis functions for input data x to be used in loops only."""
    return np.c_[poly_degree_minus_one, np.power(x, degree)]
