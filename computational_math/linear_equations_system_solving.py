import numpy as np


_ITER_LIMIT = 1000


def vec_norm(vec: np.ndarray) -> float:
    return abs(vec).max()


def Jacobi_method(a_mx: np.ndarray, err, x0_vec: np.ndarray) -> np.ndarray:
    for _ in range(_ITER_LIMIT):
        pass


def Seidel_method(a_mx: np.ndarray, err, x0_vec: np.ndarray) -> np.ndarray:
    for _ in range(_ITER_LIMIT):
        pass
