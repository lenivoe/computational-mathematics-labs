from itertools import takewhile, count
import os

import numpy as np
import matplotlib.pyplot as plt

import computational_math.runge_kutta as cmrk


DATA_DIR = 'data/' + os.path.basename(__file__)[:-3]
IMG_DIR = f'{DATA_DIR}/img'


def save_plot(img_name, title, x_vec, y_mx, a, b, desired_system):
    fig, ax = plt.subplots()

    x_points = np.linspace(a, b, int((b-a)*200))
    for i, func in enumerate(desired_system):
        ax.plot(x_points, [*map(func, x_points)], label=f'искомая y{i+1}')

    for i in range(y_mx.shape[1]):
        ax.plot(x_vec, y_mx[:, i], linewidth=2, linestyle='dashed', label=f'вычисленная y{i+1}')

    ax.set_title(title)
    ax.legend()
    fig.savefig(img_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    eq_systems_names = [
        'dy/dx = 3x^2 + (y - x^3)',
        'dy/dx = 3x^2 + 10(y - x^3)',
        'dy1/dx = y1-2*y2\n' 'dy2/dx = y1-y2-2',
        'dy1/dx = 3*y1-y2\n' 'dy2/dx = 4*y1-y2',
    ]

    eq_systems = np.array([
        [lambda x, y: 3*x**2 + (y - x**3)],
        [lambda x, y: 3*x**2 - 10*(y - x**3)],
        [lambda x, y1, y2: y1-2*y2, lambda x, y1, y2: y1-y2-2],
        [lambda x, y1, y2: 3*y1-y2, lambda x, y1, y2: 4*y1-y2]
    ])

    desired_systems = np.array([
        [lambda x: x**3],
        [lambda x: x**3],
        [lambda x: -3*np.cos(x)+5*np.sin(x)+4, lambda x: - 4*np.cos(x)+np.sin(x)+2],
        [lambda x: (5 + 2*x)*np.e**x, lambda x: (8 + 4*x)*np.e**x]
    ])

    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR)
    for fname in os.listdir(IMG_DIR):
        os.remove(f'{IMG_DIR}/{fname}')

    ifname, ofname = f'{DATA_DIR}/input.txt', f'{DATA_DIR}/output.txt'
    with open(ifname, 'r', encoding='utf-8') as reader, open(ofname, 'w', encoding='utf-8') as writer:
        it = map(str.strip, reader)
        it = filter(lambda s: s and not s.startswith('#'), it)
        it = takewhile(lambda s: s != '--break', it)

        for img_num in count(1, 2):
            try:
                num = int(next(it))
                a, b, h, err = map(float, next(it).split())

                system = eq_systems[num]
                desired = desired_systems[num]
                y0_list = np.array([func(a) for func in desired])

                print(eq_systems_names[num], file=writer)

                # с постоянным шагом
                x_vec, y_mx = cmrk.Runge_Kutta_method(system, y0_list, a, b, h)
                global_err = cmrk.global_error_norm(x_vec, y_mx, desired)

                msg = f'[{a}, {b}], h: {h},  голбальная погрешность: {global_err:.2e}'
                print(msg, file=writer)

                title = eq_systems_names[num] + '\n' + msg
                save_plot(f'{IMG_DIR}/{img_num}.png', title, x_vec, y_mx, a, b, desired)

                # с автоматическим шагом
                x_vec, y_mx = cmrk.Runge_Kutta_with_auto_step(system, y0_list, a, b, err)
                global_err = cmrk.global_error_norm(x_vec, y_mx, desired)

                msg = f'[{a}, {b}], e: {err:.1e},  голбальная погрешность: {global_err:.2e}'
                print(msg, '\n', file=writer)

                title = eq_systems_names[num] + '\n' + msg
                save_plot(f'{IMG_DIR}/{img_num+1}.png', title, x_vec, y_mx, a, b, desired)

            except StopIteration:
                break


def task5():
    system_name = 'du/dx + 30u = 0'
    system = [lambda x, y: -30*y]
    a, b = 0, 1
    y0_list = [1]
    h_list = [(b-a)/10, (b-a)/11]

    ofname = f'{DATA_DIR}/output.txt'
    with open(ofname, 'w', encoding='utf-8') as writer:
        for i, h in enumerate(h_list):
            x_vec, y_mx = cmrk.Runge_Kutta_method(system, y0_list, a, b, h)

            title = f'{system_name}\n' f'[{a}, {b}]   h: {h}\n'
            print(title, file=writer)
            save_plot(f'{IMG_DIR}/task5-{i}.png', title, x_vec, y_mx, a, b, [])

        print(file=writer)


if __name__ == '__main__':
    # main()
    task5()
