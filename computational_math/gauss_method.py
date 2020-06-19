import numpy as np


def gauss(a_mx: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    '''Метод Гаусса. a_mx - матрица из левой части СЛАУ. b_vec - столбец из правой части СЛАУ.'''

    a, b = np.copy(a_mx), np.copy(b_vec)
    if(_forwardMove(a, b)):
        return _backwardMove(a, b)
    return None


def _forwardMove(a_mx: np.ndarray, b_vec: np.ndarray) -> bool:
    for i in range(0, np.size(a_mx, 0)):
        j = i + np.argmax(abs(a_mx[i:, i]))
        if a_mx[i, j] == 0:
            return False

        if i != j:
            a_mx[(i, j), :] = a_mx[(j, i), :]
            b_vec[(i, j), ] = b_vec[(j, i), ]

        for j in range(i+1, np.size(a_mx, 0)):
            k = a_mx[j, i] / a_mx[i, i]
            a_mx[j, i:] -= k * a_mx[i, i:]
            b_vec[j] -= k * b_vec[i]

    return True


def _backwardMove(a_mx: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    x = np.empty(b_vec.size)
    for i in range(b_vec.size-1, -1, -1):
        x[i] = (b_vec[i] - a_mx[i, i+1:].dot(x[i+1:])) / a_mx[i, i]

    return x
