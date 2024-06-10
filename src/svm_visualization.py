from matplotlib import pyplot as plt
import numpy as np


def visualize_svm(X, y, w, support_vectors=None):
    """
    Vizualizuje rozhodovaciu hranicu SVM a podporné vektory.

    Parametre:
        X (numpy.ndarray): Vstupné dátové body.
        y (numpy.ndarray): Označenia vstupných dátových bodov.
        w (numpy.ndarray): Koeficienty hyperroviny.
        support_vectors (numpy.ndarray): Podporné vektory.
    """
    def get_hyperplane_value(x, w, offset):
        return (-w[1] * x + w[0] + offset) / w[2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Vykreslenie dátových bodov
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')

    # Vykreslenie rozhodovacej hranice
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, w, 0)
    x1_2 = get_hyperplane_value(x0_2, w, 0)

    x1_1_m = get_hyperplane_value(x0_1, w, -1)
    x1_2_m = get_hyperplane_value(x0_2, w, -1)

    x1_1_p = get_hyperplane_value(x0_1, w, 1)
    x1_2_p = get_hyperplane_value(x0_2, w, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    # Vykreslenie podporných vektorov
    if support_vectors is not None:
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=150, facecolors='none', edgecolors='r')

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Rozhodovacia hranica SVM')

    plt.show()