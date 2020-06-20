import numpy as np
import math


_ITER_LIMIT = 1000


def vec_norm(vec: np.ndarray) -> float:
    return abs(vec).max()


def diag_prevalence(mx):
    mx = np.abs(mx)
    return ((mx.sum(axis=1) - mx.diagonal()) / mx.diagonal()).max()


def criteria(ab_mx, err, x0_vec):
    a_mx = ab_mx[:, :-1]
    b_mx_norm = diag_prevalence(a_mx)

    jacobi_vec, i = Jacobi_method(ab_mx, np.PINF, x0_vec)
    seidel_vec, i = Seidel_method(ab_mx, np.PINF, x0_vec)

    jacobi_norm = vec_norm(jacobi_vec-x0_vec)
    seidel_norm = vec_norm(seidel_vec-x0_vec)

    k_jacobi = math.log(err * (1 - b_mx_norm) / jacobi_norm) / math.log(b_mx_norm)
    k_seidel = math.log(err * (1 - b_mx_norm) / seidel_norm) / math.log(b_mx_norm)

    return k_jacobi, k_seidel


def Jacobi_method(ab_mx: np.ndarray, err, x0_vec: np.ndarray) -> np.ndarray:
    a_mx = ab_mx[:, :-1]
    b_vec = ab_mx[:, -1]
    x_vec = x0_vec
    v_sum = np.zeros(a_mx.shape[0])

    for iter_amount in range(1, _ITER_LIMIT+1):
        for i in range(a_mx.shape[0]):
            v = (a_mx[i] * x_vec) / a_mx[i, i]
            v[i] = 0
            v_sum[i] = v.sum()

        next_x_vec = b_vec / a_mx.diagonal() - v_sum

        norm = vec_norm(next_x_vec - x_vec)
        if norm <= err:
            break

        x_vec = next_x_vec

    return next_x_vec, iter_amount


def Seidel_method(ab_mx: np.ndarray, err, x0_vec: np.ndarray) -> np.ndarray:
    a_mx = ab_mx[:, :-1]
    b_vec = ab_mx[:, -1]
    x_vec = x0_vec
    next_x_vec = x_vec.copy()

    for iter_amount in range(1, _ITER_LIMIT+1):
        for i in range(a_mx.shape[0]):
            left_sum = ((a_mx[i, :i] * next_x_vec[:i]) / a_mx[i, i]).sum()
            right_sum = ((a_mx[i, i+1:] * x_vec[i+1:]) / a_mx[i, i]).sum()
            next_x_vec[i] = b_vec[i] / a_mx[i, i] - (left_sum + right_sum)

        norm = vec_norm(next_x_vec - x_vec)
        if norm <= err:
            break

        x_vec, next_x_vec = next_x_vec, x_vec

    return next_x_vec, iter_amount
