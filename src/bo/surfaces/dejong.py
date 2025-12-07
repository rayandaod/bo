from matplotlib import pyplot as plt
import numpy as np

def dejong(X):
    """
    Args:
        X (n, d)
    Returns:
        Output of X on the dejong surface (n,)
    """
    return np.sum(X**2, axis=1)


if __name__ == '__main__':
    x1 = np.linspace(-10, 10, 21)
    x2 = np.linspace(-10, 10, 21)

    # Here X1 contains the same vector repeated along dimension 0 ([[-10, -9, ..., 9, 10], ...]),
    # and X2 contains the same vector repeated along dimension 1 ([[-10, -10, ..., -10, -10], [-9, ..., -9], ...])
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.ravel(), X2.ravel()])

    Y = dejong(X)
    Y_grid = Y.reshape(X1.shape)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y_grid, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('De Jong Function (3D Surface)')
    plt.show()
