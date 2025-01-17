import numpy as np
from implementations import least_squares
from proj1_helpers import predict_labels

def generate_prediction(x_tr_0, y_tr_0, x_tr_1, y_tr_1, x_tr_2, y_tr_2, x_tr_3, y_tr_3, x_te_0, x_te_1, x_te_2, x_te_3, jet_num_te):
    """Generate a prediction for a test dataset already split according to jet_num
    by calculating weights using a training dataset also already split."""
    #compute the weights using predetermined polynomial degrees
    w_0, _ = least_squares(y_tr_0, build_poly(x_tr_0, 9))
    w_1, _ = least_squares(y_tr_1, build_poly(x_tr_1, 15))
    w_2, _ = least_squares(y_tr_2, build_poly(x_tr_2, 13))
    w_3, _ = least_squares(y_tr_3, build_poly(x_tr_3, 12))

    #compute the prediction using the weights
    y_te_0 = predict_labels(w_0, build_poly(x_te_0, 9))
    y_te_1 = predict_labels(w_1, build_poly(x_te_1, 15))
    y_te_2 = predict_labels(w_2, build_poly(x_te_2, 13))
    y_te_3 = predict_labels(w_3, build_poly(x_te_3, 12))

    #join the four predictions into a single one matching the original indices
    predicted_y_te = []
    i_0, i_1, i_2, i_3 = 0, 0, 0, 0
    for jet_num in jet_num_te:
        if jet_num == 0:
            predicted_y_te.append(y_te_0[i_0])
            i_0 += 1
        elif jet_num == 1:
            predicted_y_te.append(y_te_1[i_1])
            i_1 += 1
        elif jet_num == 2:
            predicted_y_te.append(y_te_2[i_2])
            i_2 += 1
        else:
            predicted_y_te.append(y_te_3[i_3])
            i_3 += 1

    return predicted_y_te

def preprocess_datasets(x_tr, y_tr, x_te, y_te):
    """The pipeline used to preprocess the datasets.
    First, split the training dataset in function of jet_num, then set missing values to mean,
    remove correlated columns, standardize the dataself.
    Repeat with the training set."""
    #split the training dataset in function of the jet_num number
    x_tr_0, y_tr_0, x_tr_1, y_tr_1, x_tr_2, y_tr_2, x_tr_3, y_tr_3, _ = split_dataset_by_jet_num(x_tr, y_tr)

    #set the remaining missing values (-999) to the mean of their respective column for the training set
    x_tr_0 = set_missing_values_to_mean(x_tr_0)
    x_tr_1 = set_missing_values_to_mean(x_tr_1)
    x_tr_2 = set_missing_values_to_mean(x_tr_2)
    x_tr_3 = set_missing_values_to_mean(x_tr_3)

    #remove the correlated columns and store the corresponding indices
    x_tr_0, cols_0 = remove_correlated_columns(x_tr_0)
    x_tr_1, cols_1 = remove_correlated_columns(x_tr_1)
    x_tr_2, cols_2 = remove_correlated_columns(x_tr_2)
    x_tr_3, cols_3 = remove_correlated_columns(x_tr_3)

    #standardize the data and store the mean and std
    x_tr_0, mean_0, std_0 = standardize(x_tr_0)
    x_tr_1, mean_1, std_1 = standardize(x_tr_1)
    x_tr_2, mean_2, std_2 = standardize(x_tr_2)
    x_tr_3, mean_3, std_3 = standardize(x_tr_3)

    #split the test dataset in function of the jet_num number
    x_te_0, _, x_te_1, _, x_te_2, _, x_te_3, _, jet_num_te = split_dataset_by_jet_num(x_te, y_te)

    #set the remaining missing values (-999) to the mean of their respective column for the test set
    x_te_0 = set_missing_values_to_mean(x_te_0)
    x_te_1 = set_missing_values_to_mean(x_te_1)
    x_te_2 = set_missing_values_to_mean(x_te_2)
    x_te_3 = set_missing_values_to_mean(x_te_3)

    #delete the columns previously thought to be highly correlated in the training set for the test set
    x_te_0 = np.delete(x_te_0, cols_0, 1)
    x_te_1 = np.delete(x_te_1, cols_1, 1)
    x_te_2 = np.delete(x_te_2, cols_2, 1)
    x_te_3 = np.delete(x_te_3, cols_3, 1)

    #standardize the test set using the mean and std of the training set
    x_te_0 = (x_te_0-mean_0)/std_0
    x_te_1 = (x_te_1-mean_1)/std_1
    x_te_2 = (x_te_2-mean_2)/std_2
    x_te_3 = (x_te_3-mean_3)/std_3

    return x_tr_0, y_tr_0, x_tr_1, y_tr_1, x_tr_2, y_tr_2, x_tr_3, y_tr_3, x_te_0, x_te_1, x_te_2, x_te_3, jet_num_te

def remove_correlated_columns(x, threshold=0.9):
    """Remove the columns of a given dataset that have a correlation bigger than threshold."""
    D = x.shape[1]
    column_index = []
    correlation_matrix = np.corrcoef(x, rowvar=False)
    for i in np.arange(D):
        for j in np.arange(i):
            if correlation_matrix[i, j] >= threshold:
                column_index.append(i)

    x = np.delete(x, column_index, 1)
    return x, column_index

def split_dataset_by_jet_num(x, y):
    """According to the jet_num variable (which must be in column 22) of the dataset x,
    split x and remove columns that can't be computed depending of the value of jet_num.
    These columns are determined in the documentation of the datasetself.
    Their indices is hardcoded and must be respected."""
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
    jet_num = []
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
        jet_num.append(x[index, 22])
    x_0 = np.asarray(x_0)
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    x_3 = np.asarray(x_3)
    y_0 = np.asarray(y_0)
    y_1 = np.asarray(y_1)
    y_2 = np.asarray(y_2)
    y_3 = np.asarray(y_3)

    return x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3, jet_num

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
    """Standardize the given dataset."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x-mean)/std, mean, std

def build_poly(x, degree):
    """Polynomial basis functions for input data x."""
    poly = np.ones((len(x), 1))
    for d in (np.arange(degree) + 1):
        poly = np.c_[poly, np.power(x, d)]
    return poly
