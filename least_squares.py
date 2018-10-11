# -*- coding: utf-8 -*-
"""Exercise 3.

Least Squares Solutions
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    gram_X=np.transpose(tx)@tx
    xTy=np.transpose(tx)@y
    return np.linalg.solve(gram_X,xTy)
