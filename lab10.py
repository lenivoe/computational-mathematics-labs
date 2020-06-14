import math
from itertools import takewhile, dropwhile, accumulate
import os

import matplotlib.pyplot as plt
import numpy as np

from computational_math.utils import vectorize, scale_img
from computational_math.spline_interpolation import calc_spline_data, get_spline, get_spline_derivative
from computational_math.integral import autostep_integrate
from computational_math.monte_karlo_parallel import monte_karlo_integrate


DATA_DIR = 'data/lab10'
IMG_DIR = f'{DATA_DIR}/img'



def gen_heart_table(size):
    x_func = lambda t: 16*(np.sin(t)**3)
    y_func = lambda t: 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)

    t_vec = np.linspace(0, 2*math.pi, num=size)
    return x_func(t_vec), y_func(t_vec)

def gen_circle_table(size):
    x_func = np.cos
    y_func = np.sin

    t_vec = np.linspace(0, 2*math.pi, num=size)
    return x_func(t_vec), y_func(t_vec)


def gen_nodes_t_by_index(nodes_x, nodes_y):
    # (i, x_i), (i, y_i)
    return np.arange(len(nodes_x))

def gen_nodes_t_by_dist(nodes_x, nodes_y):
    dx = nodes_x[1:]-nodes_x[:-1]
    dy = nodes_y[1:]-nodes_y[:-1]

    # dt = sqrt((x_i1 - x_i0)^2 + (y_i1 - y_i0)^2)
    dt = np.sqrt(dx**2 + dy**2)
    return np.fromiter(accumulate((0, *dt)), float, len(dt)+1)





def main():
    ofname = f'{DATA_DIR}/output.txt'
    
    table_names = ('"Круг"', '"Сердце"', )
    table_generators = (gen_circle_table, gen_heart_table, )
    err = 10**-5
    table_size = 101

    points_amounts = (10**3, 10**4, 10**5, 10**6, )


    excess_files = set(os.listdir(IMG_DIR)) - set(map(lambda i: f'{i}.png', range(len(table_names)*len(points_amounts))))
    for fname in excess_files:
        os.remove(f'{IMG_DIR}/{fname}')

    with open(ofname, 'w', encoding='utf-8') as writer:
        i = 0
        for name, gen_table in zip(table_names, table_generators):
            for points_amount in points_amounts:
                # генерация таблицы
                x_values, y_values = gen_table(table_size)
                t_nodes = gen_nodes_t_by_index(x_values, y_values)

                # интегрирование методом Монте-Карло
                monte_karlo_sqr, x_points, y_points, is_inside = monte_karlo_integrate(x_values, y_values, points_amount)

                # создание сплайнов
                abcd_y_data = calc_spline_data(t_nodes, y_values)
                abcd_x_data = calc_spline_data(t_nodes, x_values)
                _, *bcd_dx_data = abcd_x_data
                
                y_func = get_spline(t_nodes, *abcd_y_data)
                x_func = get_spline(t_nodes, *abcd_x_data)
                dx_func = get_spline_derivative(t_nodes, *bcd_dx_data)
                

                # интегрирование формулой Симпсона
                integrand_func = vectorize(lambda t: y_func(t) * dx_func(t))
                simpson_sqr, h, iter_amount = autostep_integrate(integrand_func, t_nodes[0], t_nodes[-1], err)
                simpson_sqr = abs(simpson_sqr)


                # отображение графика
                _, ax = plt.subplots()
                
                t_vec = np.linspace(t_nodes[0], t_nodes[-1], int((t_nodes[-1]-t_nodes[0])/0.01) + 1)
                x_func, y_func = vectorize(x_func), vectorize(y_func)

                x_vec, y_vec = x_func(t_vec), y_func(t_vec)


                ax.plot(x_points[is_inside], y_points[is_inside], marker='o', ls='', markersize=1)
                is_outside = False == is_inside
                ax.plot(x_points[is_outside], y_points[is_outside], marker='o', ls='', markersize=1)
                ax.plot(x_vec, y_vec, linewidth=3)

                title = f'{name}, число точек: {points_amount:.0e}\n'
                title += f'По формуле Симпсона: {simpson_sqr}\n'
                title += f'Методом Монте-Карло: {monte_karlo_sqr}'
                ax.set_title(title)
                plt.savefig(f'{IMG_DIR}/{i}.png', bbox_inches='tight', pad_inches=0)
                
                print('test:', i)
                i += 1
    
    
    for fname in os.listdir(f'{IMG_DIR}'):
        scale_img(f'{IMG_DIR}/{fname}', 0.8)



def test():
    import computational_math.Lagrange_polynomial_interpolation as cmLi

    err = 10**-5
    table_size = 1001

    # генерация таблицы
    x_values, y_values = gen_heart_table(table_size)
    t_nodes = gen_nodes_t_by_index(x_values, y_values)

    # создание сплайнов
    abcd_y_data = calc_spline_data(t_nodes, y_values)
    abcd_x_data = calc_spline_data(t_nodes, x_values)
    _, *bcd_dx_data = abcd_x_data
    
    y_func = get_spline(t_nodes, *abcd_y_data)
    dx_func = get_spline_derivative(t_nodes, *bcd_dx_data)
    

    # интегрирование формулой Симпсона
    integrand_func = vectorize(lambda t: y_func(t) * dx_func(t))
    simpson_sqr, *_ = autostep_integrate(integrand_func, t_nodes[0], t_nodes[-1], err)
    simpson_sqr = abs(simpson_sqr)


    # отображение графика
    
    t_vec = np.linspace(t_nodes[0], t_nodes[-1], int((t_nodes[-1]-t_nodes[0])/0.01) + 1)
    ft_vec = integrand_func(t_vec)

    # Лагранж
    def foo(a,b):
        t_nodes = np.linspace(a, b, 3)
        ft_values = integrand_func(t_nodes)

        Ln = cmLi.get_Lagrange_polynomial(t_nodes, ft_values)
        Ln = vectorize(Ln)

        t = np.linspace(a, b, int(100*(a+b)/2) + 1)
        return t, Ln(t)
    
    for p in range(7):
        _, ax = plt.subplots()
        ax.plot(t_vec, ft_vec, linewidth=3)
        
        t = np.linspace(t_nodes[0], t_nodes[-1], 2*(2**p)+1)
        for a, b in zip(t[:-1], t[1:]):
            ax.plot(*foo(a,b), linewidth=2)
            
        plt.show()

        

if __name__ == '__main__':
    main()

