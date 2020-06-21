import os

import matplotlib.pyplot as plt

from computational_math.shooting_method import shooting_method
from computational_math.utils import scale_img


DATA_DIR = 'data/' + os.path.basename(__file__)[:-3]
IMG_DIR = f'{DATA_DIR}/img'


def main():
    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR)
    for fname in os.listdir(IMG_DIR):
        os.remove(f'{IMG_DIR}/{fname}')

    N, R, n = 1, 1.94, 2
    eq_system = [lambda x, y, u: u, lambda x, y, u: N*u + N*R*y**n]

    def get_first_values(y_a):
        ''' по одному из значений y(a), u(a) возвращает оба '''
        return y_a, N * (y_a - 1)

    def second_boundary_condition(y_b, u_b):
        ''' левая часть уравнения fi2( y(b), u(b) ) = 0 '''
        return u_b

    a, b = 0, 1
    err = 10**-4
    p0, p1 = 0.1, 0.7

    x_vec, y_mx = shooting_method(eq_system, a, b, err, p0, p1, get_first_values, second_boundary_condition)

    eq_system_name = "y'' - y' - 1.94y^2 = 0,\n"
    eq_system_name += "y'(0) = y(0) - 1,\n"
    eq_system_name += "y'(1) = 0"

    ofname = f'{DATA_DIR}/output.txt'
    with open(ofname, 'w', encoding='utf-8') as writer:
        print(eq_system_name, '\n', file=writer)
        print('Начальные приближения для y(0):', p0, p1, file=writer)
        print('Точность:', err, '\n', file=writer)
        print(f"{'z':>8} {'y(z)':>25} {' ':20}y'(z)", file=writer)
        data = zip(x_vec, y_mx.transpose()[0], y_mx.transpose()[1])
        print(*map(lambda d: f'{d[0]:8} {d[1]:25} {d[2]:25}', data), sep='\n', file=writer)

    fig, (ax_y, ax_u) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(eq_system_name)

    ax_y.plot(x_vec, y_mx[:, 0], label='y(z)')
    ax_y.legend()
    ax_u.plot(x_vec, y_mx[:, 1], label="y'(z)", color='orange')
    ax_u.legend()

    fig.savefig(f'{IMG_DIR}/img.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    scale_img(f'{IMG_DIR}/img.png', 1)


if __name__ == '__main__':
    main()
