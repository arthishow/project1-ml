# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e=y-tx@w
    return (-1/len(y))*np.transpose(tx)@e


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad=0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad=grad+compute_stoch_gradient(minibatch_y,minibatch_tx,w)
        loss=compute_loss(y,tx,w)
        w=w-gamma*(1/batch_size)*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws