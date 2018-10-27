import numpy as np
from costs import *
from implementations_helper import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w, loss,tol = initial_w, [], 0.01
    for n_iter in range(max_iters):
        grad, e = compute_gradient(y, tx, w)
        w = w - gamma*grad
        loss.append(compute_mse(e))
        if (np.abs(loss[n_iter]-loss[n_iter-1]))<tol and n_iter>0:
            return w, loss[-1]
    return (w, loss[-1])

#accelerated gradient descent
def acc_GD(y, tx, initial_w, max_iters, step, acc_g):
    """Linear regression using gradient descent."""
    w, loss,tol, = initial_w, [], 0.01
    grad, e0 = compute_gradient(y, tx, w)
    vt = step*grad
    loss.append(compute_mse(e0))
    for n_iter in range(max_iters):
        if n_iter > 0 :
            grad, e = compute_gradient(y, tx, w-vt)
            vt = acc_g*vt - step*grad
            loss.append(compute_mse(e))
        w = w - vt
        if (np.abs(loss[n_iter]-loss[n_iter-1]))<tol and n_iter>0:
            return w, loss[-1]
    return (w, loss[-1])

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N = y.shape[0]
    lambda_I = 2*N*lambda_ *np.identity(tx.shape[1])
    a = tx.T@tx + lambda_I
    b = tx.T@y
    w = np.linalg.solve(a, b) #solve (X_T*X + 2*N*lambda*I)*w = X_T*y and return w
    loss = compute_mse_loss(y, tx, w)
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD."""
    w, loss,tol = initial_w, [], 0.01
    for n_iter in range(max_iters):
        grad, e = compute_log_grad(y, tx, w)
        w = w - gamma*grad
        loss.append(compute_mse(e))
        if (np.abs(loss[n_iter]-loss[n_iter-1]))<tol and n_iter>0:
            return w, loss[-1]
    return (w, loss[-1])


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD."""
    raise NotImplementedError
    return (w, loss)
