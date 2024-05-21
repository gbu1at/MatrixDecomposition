import numpy
import numpy as np


def lu_decomposition(matrix: numpy.array) -> (np.array, np.array):
    n, m = matrix.shape
    assert n == m
    a00 = matrix[0, 0]

    assert a00 != 0

    if matrix.shape == (1, 1):
        return np.array([[1]]), np.array([[a00]])

    l00, w00 = (np.array([1 if i == 0 else 0 for i in range(n)]),
                np.array([a00 if i == 0 else 0 for i in range(n)]).T)

    w_u = np.array([matrix[0, 1:]])
    v_l = np.array([matrix[1:, 0] / a00]).T

    matrix_ = matrix[1:, 1:]
    L_, U_ = lu_decomposition(matrix_ - v_l.dot(w_u))
    L = np.vstack((l00, np.column_stack((v_l, L_))))
    U = np.column_stack((w00, np.vstack((w_u, U_))))

    return L, U


if __name__ == "__main__":
    ...
