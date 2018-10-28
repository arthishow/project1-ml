import numpy as np
from costs import *
from implementations_helper import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        grad, e = compute_gradient(y, tx, w)
        w = w - gamma*grad
    loss = compute_categorical_loss(y, tx, w)
    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    batch_size = 1 #as requested in the project desription
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            grad, e = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma*grad
    loss = compute_categorical_loss(y, tx, w)
    return (w, loss)

def least_squares(y, tx):
    """Least squares regression using normal equations."""
    a = tx.T@tx #Gram matrix: X_T*X
    b = tx.T@y #X_T*y
    w = np.linalg.solve(a, b) #solve (X_T*X)*w = X_T*y and return w
    loss = compute_categorical_loss(y, tx, w)
    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N = y.shape[0]
    lambda_I = 2*N*lambda_ *np.identity(tx.shape[1])
    a = tx.T@tx + lambda_I
    b = tx.T@y
    w = np.linalg.solve(a, b) #solve (X_T*X + 2*N*lambda*I)*w = X_T*y and return w
    loss = compute_categorical_loss(y, tx, w)
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        grad, e = compute_log_grad(y, tx, w)
        w = w - gamma*grad
    loss = compute_log_likelihood(y, tx, w)
    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_log_grad(y, tx, w) + 2*lambda_*w
        w = w - gamma*grad
    loss = compute_log_likelihood(y, tx, w) + lambda_*(np.linalg.norm(w)**2)
    return (w, loss)
