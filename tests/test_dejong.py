import numpy as np
from bo.surfaces.dejong import dejong


def test_dejong():
    # 1d vectors
    X_1d = np.array([[0.0], [1.0], [2.0]])
    result = dejong(X_1d)
    assert result.shape == (X_1d.shape[0],)
    assert np.all(result == (X_1d**2).flatten())

    # 3d vectors
    X_3d = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    result = dejong(X_3d)
    assert result.shape == (X_3d.shape[0],)
    assert np.all(result == np.array([5.0, 14.0]))
