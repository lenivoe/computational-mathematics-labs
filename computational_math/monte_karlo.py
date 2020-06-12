import math
import multiprocessing as mp

import numpy as np


def get_is_inside_func(x_vec, y_vec):
    '''
        Возвращает функцию, которая проверяет лежит ли точка (x,y)
        внутри многоугольинка с вершинами в (x_vec, y_vec)
    '''

    above = (x_vec - np.concatenate([x_vec[-1:], x_vec[:-1]]))
    under = (y_vec - np.concatenate([y_vec[-1:], y_vec[:-1]]))

    factors = np.empty(len(x_vec))
    factors[:] = np.NaN

    np.divide(above, under, out=factors, where=under!=0)

    del above, under

    def is_inside(x, y):
        is_in = False
        amount = 0

        for i in range(len(x_vec)):
            next_i = (i + 1) % len(x_vec)

            x_next, y_next = x_vec[next_i], y_vec[next_i]
            x_cur, y_cur = x_vec[i], y_vec[i]
            k = factors[i]
            
            if (y_next <= y and y < y_cur) or (y_cur <= y and y < y_next):
                if k != np.NaN:
                    if x > (k * (y - y_next) + x_next):
                        is_in = not is_in

        return is_in

    return is_inside


def monte_karlo(x_nodes, y_values, x_points, y_points):
    inside_amount = 0
    is_inside = get_is_inside_func(x_nodes, y_values)

    for x, y in zip(x_points, y_points):
        if is_inside(x, y):
            inside_amount += 1
    
    return inside_amount/len(x_points)



def monte_karlo_integrate(x_nodes, y_values, points_amount):
    x_min, x_max = x_nodes.min(), x_nodes.max()
    y_min, y_max = y_values.min(), y_values.max()

    x_points = np.random.sample(points_amount) * (x_max-x_min) + x_min
    y_points = np.random.sample(points_amount) * (y_max-y_min) + y_min

    part = monte_karlo(x_nodes, y_values, x_points, y_points)

    In = (x_max-x_min)*(y_max-y_min)*part
    return In, x_points, y_points
