# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    gram_X=np.transpose(tx)@tx
    lam=2*len(y)* lamb *np.identity(len(tx[1]))
    xTy=np.transpose(tx)@y
    return np.linalg.solve(gram_X+lam,xTy)
