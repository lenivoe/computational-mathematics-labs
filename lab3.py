from itertools import takewhile, dropwhile
import os
import re

import numpy as np

import computational_math.linear_equations_system_solving as less


DATA_DIR = 'data/' + os.path.basename(__file__)[:-3]


def to_numbers(s):
    return np.array([float(w) for w in re.split('[ |]', s) if w])


def gen_mx(rows_amount, cols_amount, min_val, max_val, is_symmetric=False):
    mx = np.random.sample((rows_amount, cols_amount))
    mx = np.round(mx*(max_val-min_val) + min_val)
    if is_symmetric:
        for i in range(mx.shape[0]):
            mx[:, i] = mx[i]
    return mx


def main():

    ifname, ofname = f'{DATA_DIR}/input.txt', f'{DATA_DIR}/output.txt'
    with open(ifname, 'r', encoding='utf-8') as reader, open(ofname, 'w', encoding='utf-8') as writer:
        # print(gen_mx(3, 4, -9, 10), file=writer)

        reader = reader.readlines()

        line_it = map(str.strip, reader)
        line_it = filter(lambda s: not s.startswith('#'), line_it)
        line_it = takewhile(lambda s: s != '--break', line_it)

        while True:
            try:
                it = dropwhile(lambda s: not s, line_it)
                err = float(next(it))
                x0_vec = to_numbers(next(line_it))

                it = takewhile(lambda s: s, line_it)
                ab_mx = np.array([to_numbers(s) for s in it])

                print(f'e: {err},  x0:', x0_vec, file=writer)
                print(ab_mx, '\n', file=writer)

                a_mx = ab_mx[:, :-1]
                b_vec = ab_mx[:, -1]

                b_norm = less.mx_b_norm(a_mx)
                print('Норма матрицы B:', b_norm, file=writer)
                if b_norm < 1:
                    jacobi_max_amount, seidel_max_amount = less.sufficient_condition(ab_mx, err, x0_vec)
                    print('Достаточное число итераций метода Якоби:', jacobi_max_amount, file=writer)
                    print('Достаточное число итераций метода Зейделя:', seidel_max_amount, file=writer)
                else:
                    print('Нет гарантии сходимости методов Якоби и Зейделя, так как ||B|| >= 1', file=writer)
                print(file=writer)

                x_vec, iters_amount = less.Jacobi_method(ab_mx, err, x0_vec)
                if iters_amount is np.nan:
                    print('Не удалось вычислить решение методом Якоби', file=writer)
                else:
                    print('Решение методом Якоби:', x_vec, file=writer)
                    print(f'Число итераций: {iters_amount}', file=writer)
                    print('Невязка', a_mx @ x_vec - b_vec, file=writer)
                print(file=writer)

                x_vec, iters_amount = less.Seidel_method(ab_mx, err, x0_vec)
                if iters_amount is np.nan:
                    print('Не удалось вычислить решение методом Якоби', file=writer)
                else:
                    print('Решение методом Зейделя:', x_vec, file=writer)
                    print(f'Число итераций: {iters_amount}', file=writer)
                    print('Невязка', a_mx @ x_vec - b_vec, file=writer)
                print('\n', file=writer)

            except StopIteration:
                break


if __name__ == '__main__':
    main()
