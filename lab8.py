import os
from itertools import takewhile

import matplotlib.pyplot as plt
import numpy as np

from computational_math.utils import *
import computational_math.derivative as drtv


DATA_DIR = 'data/lab8'
IMG_DIR = f'{DATA_DIR}/img'



def str_to_filename(s:str):
    return s.replace('|', ' I ').replace('/', ' div ')

def get_plot_filename(func_name, msg):
    ind = 1 + sum(1 for _ in os.listdir(IMG_DIR))
    return f'{IMG_DIR}/{ind:03} {str_to_filename(func_name)} {msg}.png'



def calc_opt_h(x_min, x_max, d_func, errors):
    uh = 10000
    nodes_amount = int((x_max-x_min)*uh)
    nodes = np.array([x_min+i/uh for i in range(nodes_amount)])
    M2 = np.max(np.abs(d_func(nodes)))

    return 2*np.sqrt(errors / M2)


def main():
    func_tests = {
        'x^4' : (
            lambda x: x**4,
            lambda x: 4*x**3,
            lambda x: 12*x**2,
            lambda x: 24*x,
        ),
        'sin(x)' : (
            np.sin,
            np.cos,
            lambda x: -np.sin(x),
            lambda x: -np.cos(x),
        ),
        'e^(-x^2)' : (
            lambda x: np.exp(-x**2),
            lambda x: -2*x*np.exp(-x**2),
            lambda x: (4*x**2 - 2)*np.exp(-x**2),
            lambda x: (-8*x**3 + 12*x)*np.exp(-x**2),
        ),
    }

    # интерполированные функции производных для таблицы значений функции
    int_d_func_list = (
        drtv.calc_first_derivative_values,
        drtv.calc_second_derivative_values,
        drtv.calc_third_derivative_values,
    )

    # границы индексов для узлов
    # нужны, чтобы отсечь те значения, которых не будет в интерполированных производных
    borders = ((0,None), (1,-1), (2,-2),)

    # инициализация генератора псевдослучайных чисел для воспроизводимости тестов
    np.random.seed(0)
    
    ifname = f'{DATA_DIR}/input.txt'
    with open(ifname, 'r') as reader:
        for fname in os.listdir(IMG_DIR):
            os.remove(f'{IMG_DIR}/{fname}')

        reader_it = takewhile(lambda s: s != '--break\n', reader)
        reader_it = filter(lambda s: not s.startswith('#'), reader_it)
        reader_it = map(str.strip, reader_it)
        reader_it = filter(lambda s: s != '', reader_it)

        while True:
            try:
                func_name = next(reader_it)
                # функция и ее производные от первой до третьей включительно
                func, *d_func_list = func_tests[func_name]
                x_min, x_max = map(float, next(reader_it).split())
                h_list = [*map(float, next(reader_it).split())]
                errors = np.array([*map(float, next(reader_it).split())])

                #print(calc_opt_h(x_min, x_max, d_func_list[1], np.array(sorted({*errors}))))

                for h, err in zip(h_list, errors):
                    nodes_amount = int((x_max-x_min)/h)
                    nodes = np.array([x_min+i*h for i in range(nodes_amount)])

                    values = func(nodes)

                    # три списка узлов с разным количеством под первую, вторую и третью производные функции
                    d_nodes_list = [nodes[lt:rt] for lt, rt in borders]

                    # значения функции с погрешностью
                    val_errors = np.random.sample(len(nodes)) * err
                    corrupted_values = values + val_errors

                    # список значений производных функции порядка от первого до третьего в узловых точках
                    d_values_list = [func(nodes) for func, nodes in zip(d_func_list, d_nodes_list)]
                    
                    # список значений интерполированных производных функции
                    # порядка от первого до третьего в узловых точках
                    int_d_values_list = [d_func(corrupted_values, h) for d_func in int_d_func_list]


                    # вывод графиков
                    _, ax_list = plt.subplots(2,2, figsize=(9, 8))
                    ax_u, *ax_du_list = ax_list.flatten()

                    ax_u.plot(nodes, values, label='u(x_i)')
                    ax_u.plot(nodes, corrupted_values, label=f'u(x_i)')
                    ax_u.set_title(f'u(x_i)={func_name}, h={h}, err={err}')
                    ax_u.legend()

                    for i, name in enumerate(('du', 'd2u', 'd3u')):
                        ax = ax_du_list[i]
                        d_nodes = d_nodes_list[i]
                        d_values, int_d_values = d_values_list[i], int_d_values_list[i]

                        ax.plot(d_nodes, d_values, label=f'{name}(x_i)')
                        ax.plot(d_nodes, int_d_values, label=f'{name}(x_i) interpolated')

                        max_err = np.max(np.abs(d_values - int_d_values))
                        max_err = round(max_err, 6)
                        
                        ax.set_title(f'{name}(x_i), max err={max_err}')
                        ax.legend()

                    plt.subplots_adjust(wspace=0.05)
                    
                    img_name = get_plot_filename(func_name, f'h = {h}, err = {err}')
                    plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
                    scale_img(img_name, 0.7)
                    
                    #plt.show()
                    plt.close('all')

            except StopIteration:
                break
        

if __name__ == '__main__':
    main()


