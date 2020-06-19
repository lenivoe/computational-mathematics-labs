from itertools import takewhile
import os

import numpy as np
import sympy as sy

import computational_math.newtons_system as cmns


DATA_DIR = 'data/' + os.path.basename(__file__)[:-3]
IMG_DIR = f'{DATA_DIR}/img'


def draw_plots():
    x, y = sy.symbols('x, y')
    func1 = sy.Or(sy.Eq(x**2 + y**2 - 4, 0), sy.Eq(x**4 - y - 4, 0))
    func2 = sy.Or(sy.Eq((x-1)**2 + y**2 - 9, 0), sy.Eq(sy.tan(x-1) - y**3, 0))
    func3 = sy.Or(sy.Eq(x**2 + y**2 - 4, 0), sy.Eq(x**2 + y**2 - 1, 0))

    sy.plot_implicit(func1)
    sy.plot_implicit(func2, (x, -4, 6), (y, -5, 5))
    sy.plot_implicit(func3)


def main():
    eq_systems_names = [
            'x^2 + y^2 - 4 = 0\n'
            'x^4 - y - 4 = 0',

            '(x-1)^2 + y^2 - 9 = 0\n'
            'tg(x-1) - y^3 = 0',

            'x^2 + y^2 - 4 = 0\n'
            'x^2 + y^2 - 1 = 0',
    ]

    eq_systems = np.array([
        [
            lambda x, y: x**2 + y**2 - 4,
            lambda x, y: x**4 - y - 4,
        ],
        [
            lambda x, y: (x-1)**2 + y**2 - 9,
            lambda x, y: np.tan(x-1) - y**3,
        ],
        [
            lambda x, y: x**2 + y**2 - 4,
            lambda x, y: x**2 + y**2 - 1,
        ],
    ])

    df_matricies = np.array([
        [
            [lambda x, y: 2*x, lambda x, y: 2*y],
            [lambda x, y: 4*x**3, lambda x, y: -1],
        ],
        [
            [lambda x, y: 2*(x-1), lambda x, y: 2*y],
            [lambda x, y: 1/np.cos(x-1)**2, lambda x, y: -3*y**2],
        ],
        [
            [lambda x, y: 2*x, lambda x, y: 2*y],
            [lambda x, y: 2*x, lambda x, y: 2*y],
        ],
    ])

    ifname, ofname = f'{DATA_DIR}/input.txt', f'{DATA_DIR}/output.txt'
    with open(ifname, 'r', encoding='utf-8') as reader, open(ofname, 'w', encoding='utf-8') as writer:
        line_it = map(str.strip, reader)
        line_it = filter(lambda s: not s.startswith('#'), line_it)
        line_it = takewhile(lambda s: s != '--break', line_it)

        while True:
            it = takewhile(lambda s: s, line_it)

            try:
                sys_num = int(next(it))

                errors = [*map(float, next(it).split())]
                errors = sorted(errors, key=abs, reverse=True)
                errors = np.array(errors, float)
            except StopIteration:
                break

            x0_vec_list = np.array([[*map(float, s.split())] for s in it])

            sys_name = eq_systems_names[sys_num]
            eq_system = eq_systems[sys_num]
            df_mx = df_matricies[sys_num]

            print(sys_name, '\n', file=writer)
            for x0_vec in x0_vec_list:
                for err in errors:
                    x0_str = ', '.join(f'{x:g}' for x in x0_vec)
                    print(f'точность: {err:g},  (x0, y0): ({x0_str})', file=writer)

                    (x, y), iters_amount = cmns.find_root(x0_vec, eq_system, df_mx, err)
                    if iters_amount == cmns.LIMIT:
                        fmt = f'решение: ({x:g}, {y:g}), метод не сошелся'
                        print(fmt, file=writer)

                        residual = np.array([eq(x, y) for eq in eq_system])
                        print(f'невязка: {residual}\n', file=writer)

                        break

                    fmt = f'решение: ({x:g}, {y:g}), число шагов: {iters_amount}'
                    print(fmt, file=writer)

                    residual = np.array([eq(x, y) for eq in eq_system])
                    print(f'невязка: {residual}\n', file=writer)

            print(file=writer)


if __name__ == '__main__':
    # draw_plots()
    main()
