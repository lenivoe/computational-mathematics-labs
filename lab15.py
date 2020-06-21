import os

import numpy as np
from matplotlib import pyplot as plt

import computational_math.tridiagonal_matrix_algorithm as tma


DATA_DIR = 'data/' + os.path.basename(__file__)[:-3]
IMG_DIR = f'{DATA_DIR}/img'


def calc_ABCD_vec(A, B, C, a, b, c, d, lt, rt, n):
    h = (rt - lt) / n
    x_vec = np.linspace(lt, rt, n)[1:]

    a_vec = [a[0], *[1 - A(x) * h / 2 for x in x_vec], a[1]]
    b_vec = [b[0], *[2 - B(x) * h**2 for x in x_vec], b[1]]
    c_vec = [c[0], *[1 + A(x) * h / 2 for x in x_vec], c[1]]
    d_vec = [d[0], *[C(x) * h**2 for x in x_vec], d[1]]

    return zip(a_vec, b_vec, c_vec, d_vec)


def calc_first_approx(lt, rt, A, B, C, F1, F2, D1, D2, E1, E2, n):
    h = (rt - lt) / n

    a0 = 0
    b0 = D1 - h * F1
    c0 = D1
    d0 = h * E1

    an = -D2
    bn = -D2 - h * F2
    cn = 0
    dn = h * E2

    vec_ABCD = calc_ABCD_vec(A, B, C, (a0, an), (b0, bn), (c0, cn), (d0, dn), lt, rt, n)

    return abs(np.array(tma.calc(vec_ABCD)))


def calc_second_approx(lt, rt, A, B, C, F1, F2, D1, D2, E1, E2, n):
    h = (rt - lt) / n

    a0 = 0
    b0 = -F1 * h + D1 + D1 * (A(lt) - B(lt) * h) * (h / 2)
    c0 = A(lt) * D1 * h / 2 + D1
    d0 = E1 * h + C(lt) * D1 * h**2 / 2

    an = A(rt) * D2 * h / 2 - D2
    bn = (-F2 * h) - D2 + (D2 * (A(rt) + B(rt) * h) * (h / 2))
    cn = 0
    dn = E2 * h - C(rt) * D2 * h**2 / 2

    vec_ABCD = calc_ABCD_vec(A, B, C, (a0, an), (b0, bn), (c0, cn), (d0, dn), lt, rt, n)

    return abs(np.array(tma.calc(vec_ABCD)))


def C_norm_by_values(y_vec):
    return abs(y_vec).max()


def main():
    def A(x): return 2/x
    def B(x): return -1
    def C(x): return 0

    lt, rt = 1, 2
    F1, F2 = 1, 3/2
    D1, D2 = 1, 1
    E1, E2 = -1/np.e, 0

    def fst_approx(n):
        return calc_first_approx(lt, rt, A, B, C, F1, F2, D1, D2, E1, E2, n)

    def snd_approx(n):
        return calc_second_approx(lt, rt, A, B, C, F1, F2, D1, D2, E1, E2, n)

    def u_func(x):
        return np.exp(-x)/x

    parts_amounts = [25, 50, 100, 200]

    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR)
    for fname in os.listdir(IMG_DIR):
        os.remove(f'{IMG_DIR}/{fname}')

    ofname = f'{DATA_DIR}/output.txt'
    with open(ofname, 'w', encoding='utf-8') as writer:
        for n in parts_amounts:
            x_vec = np.linspace(lt, rt, n+1)
            u_values = u_func(x_vec)

            fst_approx_values = fst_approx(n)
            snd_approx_values = snd_approx(n)

            print(f'Число частей: {n}', file=writer)
            fst_norm = C_norm_by_values(fst_approx_values - u_values)
            print(f'Норма разности с решением первого порядка аппроксимации: {fst_norm}', file=writer)
            snd_norm = C_norm_by_values(snd_approx_values - u_values)
            print(f'Норма разности с решением второго порядка аппроксимации: {snd_norm}', file=writer)

            # plot
            fig, ax = plt.subplots()
            ax.set_title(f'Решение с разбиением n = {n}.')

            ax.plot(x_vec, u_values, label='u(x)', linewidth=2)
            ax.plot(x_vec, fst_approx_values, label='u(x) approximated #1')
            ax.plot(x_vec, snd_approx_values, label='u(x) approximated #2')

            ax.legend()
            fig.savefig(f'{IMG_DIR}/{n}.png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
