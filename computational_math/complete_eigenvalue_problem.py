import numpy as np


def _get_iters_amount(r_vec, err):
    LIMIT = 10**3

    size = r_vec.shape[0]
    err_reducer = 1 - 2/(size*(size-1))
    cur_err = r_vec.sum() * err_reducer

    for amount in range(1, LIMIT):
        cur_err *= err_reducer
        if cur_err < err:
            break
    return amount


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

    iters_amount = _get_iters_amount(r_vec, err)
    for _ in range(iters_amount):
        # вычисление индексов k, l
        k = np.argmax(r_vec)
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
