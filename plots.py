import numpy as np
import matplotlib.pyplot as plt

def cross_validation_visualization(k_fold, degrees, loss_tr, loss_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.plot(degrees, loss_tr, marker=".", color='b', label='Train error')
    plt.plot(degrees, loss_te, marker=".", color='r', label='Test error')
    plt.xlabel("Degree")
    plt.ylabel("Loss")
    plt.title("{k}-fold cross validation".format(k=k_fold))
    plt.legend(loc=2)
    plt.grid(True)

def bias_variance_decomposition_visualization(degrees, loss_tr, loss_te):
    """visualize the bias variance decomposition."""
    loss_tr_mean = np.expand_dims(np.mean(loss_tr, axis=0), axis=0)
    loss_te_mean = np.expand_dims(np.mean(loss_te, axis=0), axis=0)
    plt.plot(
        degrees,
        loss_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.3)
    plt.plot(
        degrees,
        loss_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.3)
    plt.plot(
        degrees,
        loss_tr_mean.T,
        'b',
        linestyle="-",
        label='Train (average)',
        linewidth=3)
    plt.plot(
        degrees,
        loss_te_mean.T,
        'r',
        linestyle="-",
        label='Test (average)',
        linewidth=3)
    plt.xlabel("Degree")
    plt.ylabel("Error")
    plt.legend(loc=2)
    plt.title("Bias-Variance Decomposition")
