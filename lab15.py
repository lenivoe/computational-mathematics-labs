import numpy as np
import computational_math.tridiagonal_matrix_algorithm as tma


def get_lr():
    l, r = 1, 2
    return l, r


def get_condition():
    l, r = get_lr()

    F1, F2 = 1, 3/2
    D1, D2 = 1, 1
    E1, E2 = -1/np.e, 0

    return l, r, F1, F2, D1, D2, E1, E2


def get_ABC():
    def A(x): return 2/x
    def B(x): return -1
    def C(x): return 0

    return A, B, C


def get_u_vec(n):
    l, r = get_lr()
    def u(x): return (np.e**(-x))/x
    return ((u(x) for x in [np.arange(l, r, h) + [r]]))


def calc_ABCD_vec(a, b, c, d, h, l, r):
    x_vec = np.arange(l + h, r, h)
    A, B, C = get_ABC()

    a_vec = [a[0], *[1 - A(x) * h / 2 for x in x_vec], a[1]]
    b_vec = [b[0], *[2 - B(x) * h**2 for x in x_vec], b[1]]
    c_vec = [c[0], *[1 + A(x) * h / 2 for x in x_vec], c[1]]
    d_vec = [d[0], *[C(x) * h**2 for x in x_vec], d[1]]

    return (a_vec, b_vec, c_vec, d_vec)


def first_aprox(n):
    l, r, F1, F2, D1, D2, E1, E2 = get_condition()

    h = abs(r - l) / n

    a0, an = 0, -D2
    b0, bn = D1 - h * F1, -D2 - h * F2
    c0, cn = D1, 0
    d0, dn = h * E1, h*E2

    vec_ABCD = calc_ABCD_vec((a0, an), (b0, bn), (c0, cn), (d0, dn), h, l, r)

    return tma.calc(vec_ABCD)


def second_aprox(n):
    l, r, F1, F2, D1, D2, E1, E2 = get_condition()
    A, B, C = get_ABC()
    h = abs(r - l) / n

    a0 = 0
    b0 = -F1 * h + D1 + D1 * (A(l) - B(l) * h) * (h / 2)
    c0 = A(l) * D1 * h / 2 + D1
    d0 = E1 * h + C(l) * D1 * h**2 / 2

    an = A(r) * D2 * h / 2 - D2
    bn = (-F2 * h) - D2 + (D2 * (A(r) + B(r) * h) * (h / 2))
    cn = 0
    dn = E2 * h - C(r) * D2 * h**2 / 2

    vec_ABCD = calc_ABCD_vec((a0, an), (b0, bn), (c0, cn), (d0, dn), h, l, r)

    return tma.calc(vec_ABCD)

# TODO эту я совсем не меняла


def c_norm(y1, y2):
    return max(map(lambda y: abs(y[0] - y[1]), zip(y1, y2)))


def main():
    cells = [25, 50, 100, 200]
    for n in cells:
        result_fa = first_aprox(n)
        result_sa = second_aprox(n)

        result_orig = get_u_vec(n)

        norm_first = c_norm(result_fa, result_orig)
        norm_sec = c_norm(result_sa, result_orig)
