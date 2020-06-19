import numpy as np

from computational_math.gauss_method import gauss


LIMIT = 10**3


def _norm(x):
    return np.sqrt(np.sum(x**2))


def find_root(x0_vec, eq_system, df_mx, err):
    x_vec = x0_vec.copy()
    for iters_amount in range(1, LIMIT+1):
        gen = (df(*x_vec) for df in df_mx.flatten())
        jacobi_mx = np.fromiter(gen, float, df_mx.size).reshape(*df_mx.shape)

        gen = (-eq(*x_vec) for eq in eq_system)
        fx_vec = np.fromiter(gen, float, eq_system.size)

        delta_x = gauss(jacobi_mx, fx_vec)
        if delta_x is None:
            break

        x_vec += delta_x

        if _norm(delta_x) < err:
            return x_vec, iters_amount

    return x_vec, LIMIT
