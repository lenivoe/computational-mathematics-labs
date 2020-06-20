from itertools import takewhile
import os

import numpy as np

from computational_math.complete_eigenvalue_problem import jacobi_method, residual


DATA_DIR = 'data/' + os.path.basename(__file__)[:-3]


def main():
    ifname, ofname = f'{DATA_DIR}/input.txt', f'{DATA_DIR}/output.txt'
    with open(ifname, 'r', encoding='utf-8') as reader, open(ofname, 'w', encoding='utf-8') as writer:
        line_it = map(str.strip, reader)
        line_it = filter(lambda s: not s.startswith('#'), line_it)
        line_it = takewhile(lambda s: s != '--break', line_it)

        print('используется норма невязки вектора: ||a|| = max(|a_i|)\n', file=writer)
        while True:
            lines = takewhile(lambda s: s, line_it)

            try:
                errors = np.array([*map(float, next(lines).split())])
            except StopIteration:
                break

            a_mx = np.array([[*map(float, line.split())] for line in lines])
            print('матрица:\n', a_mx, '\n', file=writer)

            for err in errors:
                values, vectors, amount = jacobi_method(a_mx, err)
                res = residual(a_mx, values, vectors)

                print(f'погрешность: {err},  число итераций: {amount}', file=writer)
                print('список норм векторов невязок:', res, file=writer)
                print('собственные числа:', values, file=writer)
                print('собственные вектора:', file=writer)
                for i in range(vectors.shape[1]):
                    print(vectors[:, i], file=writer)
                print(file=writer)

            print(file=writer)


def gen_mx(size):
    mx = np.round(np.random.sample((size, size))*100).astype(int)
    for i in range(mx.shape[0]):
        mx[:, i] = mx[i]
    return mx


if __name__ == '__main__':
    main()
