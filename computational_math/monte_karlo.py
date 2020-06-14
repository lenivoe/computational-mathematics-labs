import math
import multiprocessing as mp

import numpy as np

import computational_math.spline_interpolation as cmsi


def max_ordered(vec, pred):
    for i, (a, b) in enumerate(zip(vec[:-1], vec[1:])):
        if not pred(a, b):
            return i
    return len(vec)-1

def generate_spline_list(nodes, values):
    spline_list = []

    i = 0
    while len(nodes) > 1:
        less = max_ordered(nodes, np.less)
        greater = max_ordered(nodes, np.greater)
        
        if less >= greater:
            sub_nodes, sub_values = nodes[:less+1], values[:less+1]
        else:
            sub_nodes, sub_values = nodes[greater::-1], values[greater::-1]
            
        assert len(sub_nodes) > 1 and sub_nodes[0] < sub_nodes[1]
        
        i = max(less, greater)
        values, nodes = values[i:], nodes[i:]
        
        if not (len(sub_nodes) == 2 and abs(sub_nodes[0]-sub_nodes[-1]) <= 0):
            spline = cmsi.get_spline(sub_nodes, *cmsi.calc_spline_data(sub_nodes, sub_values))

            # добавляется (<первый узел>, <последний узел>, <сплайн>)
            spline_list.append((*sub_nodes[[0, -1]], spline))

    
    return spline_list


def get_is_inside_func(x_vec, y_vec):
    '''
        Возвращает функцию, которая проверяет лежит ли точка (x,y)
        внутри многоугольинка с вершинами в (x_vec, y_vec)
    '''

    splines = generate_spline_list(y_vec, x_vec)

    def is_inside(x, y):
        is_in = False

        for y_min, y_max, spline in splines:
            if y_min <= y and y <= y_max:
                if x > spline(y):
                    is_in = not is_in

        return is_in

    return is_inside



def monte_karlo(x_nodes, y_values, x_points, y_points):
    is_inside = get_is_inside_func(x_nodes, y_values)

    is_inside_list = (is_inside(x, y) for x, y in zip(x_points, y_points))
    is_inside_list = np.fromiter(is_inside_list, bool, len(x_points))

    return sum(is_inside_list)/len(x_points), is_inside_list



def monte_karlo_integrate(x_nodes, y_values, points_amount):
    x_min, x_max = x_nodes.min(), x_nodes.max()
    y_min, y_max = y_values.min(), y_values.max()

    x_points = np.random.sample(points_amount) * (x_max-x_min) + x_min
    y_points = np.random.sample(points_amount) * (y_max-y_min) + y_min

    ratio, is_inside_list = monte_karlo(x_nodes, y_values, x_points, y_points)

    In = (x_max-x_min)*(y_max-y_min)*ratio
    return In, x_points, y_points, is_inside_list
