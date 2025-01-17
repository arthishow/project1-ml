import numpy as np
from costs import *
from proj1_helpers import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = np.shape(y)[0]
    e = y - predict_labels(w,tx)
    grad = -(1/N)*(tx.T)@e
    return grad, e

def compute_log_grad(y,tx,w)
    sig = np.divide(np.exp(tx@w),1+np.exp(tx@w))
    grad = tx.T*(sig-y)
    loss = np.sum(np.log(1+np.exp(tx@w))-y*(tx@w))
    return grad, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
