import numpy as np


LIMIT = 10**3


def _mul_by_u_mx(mx, k, t, alpha, beta):
    k_vec = mx[:, k]*alpha + mx[:, t]*beta
    t_vec = -mx[:, k]*beta + mx[:, t]*alpha
    mx[:, k], mx[:, t] = k_vec, t_vec


def jacobi_method(a_mx, err):
    # инициализация вектора сумм r
    sqr_a_mx = a_mx**2
    np.fill_diagonal(sqr_a_mx, 0)
    r_vec = sqr_a_mx.sum(axis=1)

    b_mx = a_mx.copy()
    d_mx = np.identity(a_mx.shape[0])

    for iters_amount in range(LIMIT):
        k = np.argmax(r_vec)

        if r_vec[k] < err*10**-2:  # оценка вклада недиагональных элементов
            break

        row = abs(b_mx[k])
        row[k] = 0
        t = np.argmax(row)  # t - это l

        # вычисление alpha и beta
        if b_mx[k, k] == b_mx[t, t]:
            alpha = beta = 1/2**(1/2)
        else:
            mu = 2*b_mx[k, t]/(b_mx[k, k] - b_mx[t, t])
            fract = 0.5/np.sqrt(1+mu**2)

            alpha = np.sqrt(0.5 + fract)
            beta = np.sign(mu) * np.sqrt(0.5 - fract)

        # домножение матриц b и d на матрицу u
        _mul_by_u_mx(d_mx, k, t, alpha, beta)
        _mul_by_u_mx(b_mx, k, t, alpha, beta)
        _mul_by_u_mx(b_mx.transpose(), k, t, alpha, beta)

        # обновление вектора сумм r
        sqr_r = b_mx[[k, t], :]**2
        sqr_r[np.arange(2), [k, t]] = 0
        r_vec[[k, t]] = sqr_r.sum(axis=1)

    return b_mx.diagonal().copy(), d_mx, iters_amount+1


def residual(a_mx, eigenvalues, eigenvectors):
    return abs(a_mx @ eigenvectors - eigenvalues * eigenvectors).max(axis=0)
