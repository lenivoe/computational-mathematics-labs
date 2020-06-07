import math
from itertools import accumulate, chain
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import os

from computational_math.utils import *
import computational_math.Lagrange_polynomial_interpolation as Lpi
import computational_math.spline_interpolation as si


DATA_DIR = 'data/lab7'
IMG_DIR = f'{DATA_DIR}/img'


def plot(*func_list, labels=None, title='', x_min=-1, x_max=1, need_show=True, img_name=None):
    '''
        Отображает графики указанных функий функций
        - func_list - список функций для отображаения на графике
        - labels - необязательные метки графиков функций
        - title - заголовок
        - x_min - начало отрезка по оси X
        - x_max - конец отрезка по оси X
        - need_show - нужно ли вывести окно с графиками
        - img_name - если задан путь к файлу, туда будет сохранено изображение графиков
    '''

    assert img_name and not img_name.isspace() or need_show, 'plot func must output something'
    
    if labels == None:
        labels = (None, )*len(func_list)

    _, ax = plt.subplots()

    x = np.arange(x_min, x_max, 0.01)
    for func, label in zip(func_list, labels):
        ax.plot(x, func(x), label=label)

    ax.set_title(title)
    ax.legend()
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98, top=0.94, wspace=0.11, hspace=0.11)
    
    if img_name:
        plt.savefig(img_name)
    if need_show:
        plt.show()
    plt.close('all')


def to_nums(s):
    '''Конвертирует строку в генератор по числам'''
    return map(float, s.split())


def str_to_filename(s:str):
    return s.replace('|', ' I ').replace('/', ' div ')

def get_plot_filename(test_mode, func_name, nodes_x):
    amount = reduce(lambda sum, _: sum+1, os.listdir(f'{IMG_DIR}'), 1)
    return f'{IMG_DIR}/{amount:03} {test_mode} {str_to_filename(func_name)} [{" ".join(map(str, nodes_x))}].png'


def main():
    func_tests = {
        '|x|'        : np.abs,
        'e^(-x^2)'   : lambda x: np.exp(-x*x),
        'sin(x)'     : np.sin,
        'x^(4/3)'    : vectorize(lambda x: power(x, 4, 3)),
        'e^(x^4)'    : lambda x: np.exp(x**4),
        'x^3'        : lambda x: x**3,
        'x^9'        : lambda x: x**9,
    }

    ifname = f'{DATA_DIR}/input.txt'
    with open(ifname, 'r') as reader:
        for fname in os.listdir(f'{IMG_DIR}'):
            os.remove(f'{IMG_DIR}/{fname}')

        reader_it = iter(s.strip() for s in reader if not s.isspace() and not s.startswith('#'))

        lines_it = reader_it
        test_type = '--standard'

        while True:
            try:
                fst_line = next(reader_it)
                if fst_line.startswith('--'):
                    test_type = fst_line
                else:
                    lines_it = chain((fst_line,), reader_it)

                if test_type == '--standard':
                    standard_test(lines_it, func_tests)

                elif test_type == '--errors':
                    errors_test(lines_it, func_tests)

                elif test_type == '--parametric':
                    test_parametric_func(int(next(lines_it)))

                elif test_type == '--break':
                    break

                else:
                    raise SyntaxError('unresolved command')

            except StopIteration:
                break
        
def standard_test(lines_it, func_tests):
    func_name = next(lines_it)
    func = func_tests[func_name]

    nodes_x = np.fromiter(to_nums(next(lines_it)), float)
    nodes_y = func(nodes_x)

    assert np.array_equal(np.array(sorted(set(nodes_x))), nodes_x), 'nodes must be ordered without repetitions'

    L_n = Lpi.getInterpolatedFunc(nodes_x, nodes_y)
    S = si.getInterpolatedFunc(nodes_x, nodes_y)
    L_n, S = vectorize(L_n), vectorize(S)

    title = f'f(x)={func_name},  узлы: {nodes_x}'
    
    labels = (func_name, 'Lagrange', 'spline')
    image_name = get_plot_filename('standard', func_name, nodes_x)
    plot(func, L_n, S, labels=labels, title=title, x_min=nodes_x[0], x_max=nodes_x[-1], need_show=False, img_name=image_name)

def errors_test(lines_it, func_tests):
    func_name = next(lines_it)
    func = func_tests[func_name]

    nodes_x = np.fromiter(to_nums(next(lines_it)), dtype=float)
    nodes_y = func(nodes_x)

    i = int(next(lines_it))
    errors = (*to_nums(next(lines_it)), )

    y = nodes_y[i]

    S = si.getInterpolatedFunc(nodes_x, nodes_y)
    func_list = [func, vectorize(S)]

    for err in errors:
        nodes_y[i] = y + err
        S = si.getInterpolatedFunc(nodes_x, nodes_y)
        func_list.append(vectorize(S))

    title = f'f(x)={func_name},  узлы: {nodes_x}, ошибка в: {nodes_x[i]}'
    labels = (func_name, 'error: 0', *(f'error: {err}' for err in errors))
    image_name = get_plot_filename('errors', func_name, nodes_x)
    plot(*func_list, labels=labels, title=title, x_min=nodes_x[0], x_max=nodes_x[-1], need_show=False, img_name=image_name)

def test_parametric_func(list_len):
    # heart
    heart_name = 'heart'
    heart_x = lambda t: 16*(np.sin(t)**3)
    heart_y = lambda t: 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    heart_t = np.linspace(0, 2*math.pi, num=list_len)

    # e^(-x^2)
    exp_name = 'e^(-x^2)'
    exp_x = lambda x: x
    exp_y = lambda x: np.exp(-x**2)
    exp_t = np.linspace(-4, 4, num=list_len)

    data = zip([heart_name, exp_name], [heart_x, exp_x], [heart_y, exp_y], [heart_t, exp_t])
    for func_name, func_x, func_y, init_t in data:
        nodes_x, nodes_y = func_x(init_t), func_y(init_t)
        dx = nodes_x[1:]-nodes_x[:-1]
        dy = nodes_y[1:]-nodes_y[:-1]

        # (i, x_i), (i, y_i)
        nodes_t = np.arange(len(nodes_x))
        nodes_t_list = [nodes_t]

        # dt = sqrt((x_i1 - x_i0)^2 + (y_i1 - y_i0)^2)
        dt = np.sqrt(dx**2 + dy**2)
        nodes_t = np.fromiter(accumulate((0, *dt)), float)
        nodes_t_list.append(nodes_t)

        # dt = max(|x_i|, |y_i|)
        dt = (max(abs(x), abs(y)) for x, y in zip(nodes_x, nodes_y))
        nodes_t = np.fromiter(accumulate(dt), float)
        nodes_t_list.append(nodes_t)

        # dt = |acos(x_i1/sqrt(x_i1^2 + y_i1^2)) - acos(x_i0/sqrt(x_i0^2 + y_i0^2))|
        dt = np.arccos(nodes_x/np.sqrt(nodes_x**2 + nodes_y**2))
        dt = abs(dt[1:] - dt[:-1])
        nodes_t = np.fromiter(accumulate((0, *dt)), float)
        nodes_t_list.append(nodes_t)

        
        nrows = math.ceil(len(nodes_t_list)/2)
        ncols = math.floor(len(nodes_t_list)/2)
        size = int(len(nodes_t_list)*2.5)
        _, axes = plt.subplots(nrows, ncols, figsize=(size, size))
        axes = axes.flatten()

        for nodes_t, ax in zip(nodes_t_list, axes):
            vec_t = np.arange(init_t[0], init_t[-1], 0.01)
            ax.plot(func_x(vec_t), func_y(vec_t), label='heart')

            int_x = si.getInterpolatedFunc(nodes_t, nodes_x)
            int_y = si.getInterpolatedFunc(nodes_t, nodes_y)
            int_x, int_y = vectorize(int_x), vectorize(int_y)

            vec_t = np.arange(nodes_t[0], nodes_t[-1], 0.01)
            ax.plot(int_x(vec_t), int_y(vec_t), label='spline')

        titles = (
            f'{func_name}: (i, x_i), (i, y_i)',
            'dt = sqrt((x_i1 - x_i0)^2 + (y_i1 - y_i0)^2)',
            'dt = max(|x_i|, |y_i|)',
            'dt = delta(acos(x_i/sqrt(x_i^2 + y_i^2)))',
        )

        for ax, title in zip(axes, titles):
            ax.set_title(title)
            ax.legend()

        plt.subplots_adjust(left=0.05, right=0.985, bottom=0.06, top=0.94, wspace=0.11, hspace=0.11)
        plt.savefig(get_plot_filename('parametric', func_name, [list_len]))
        #plt.show()
        plt.close('all')



if __name__ == '__main__':
    main()


