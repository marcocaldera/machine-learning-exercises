import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html

def plotMargin(_clf, X, y, title=""):
    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = _clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    plt.title(title)
    if (len(_clf.support_vectors_) != 0):
        plt.scatter(_clf.support_vectors_[:, 0], _clf.support_vectors_[:, 1], facecolors='none', edgecolors='b',
                    linewidths=3, s=120)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
