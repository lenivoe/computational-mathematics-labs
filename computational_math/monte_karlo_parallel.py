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
            spline_data = cmsi.calc_spline_data(sub_nodes, sub_values)

            # добавляется (<первый узел>, <коэффициенты сплайна>)
            spline_list.append((*sub_nodes[[0, -1]], (sub_nodes, *spline_data)))

    
    return spline_list


def is_inside(x, y, spline_data):
    is_in = False

    for y_min, y_max, data in spline_data:
        if y_min <= y and y <= y_max:
            if x > cmsi.spline(y, *data):
                is_in = not is_in

    return is_in


def calc_worker(args):
    x_points, y_points, check_data = args

    is_inside_list = (is_inside(x, y, check_data) for x, y in zip(x_points, y_points))
    is_inside_list = np.fromiter(is_inside_list, bool, len(x_points))

    return is_inside_list


def monte_karlo(x_nodes, y_values, x_points, y_points):
    check_data = generate_spline_list(y_values, x_nodes)
    
    cpu = mp.cpu_count()
    job_size = math.ceil(len(x_points)/cpu)
    ab_list = zip(range(0, cpu*job_size, job_size), range(job_size, (cpu+1)*job_size, job_size))

    data = ((x_points[a:b], y_points[a:b], check_data) for a, b in ab_list)

    with mp.Pool(cpu) as pool:
        is_inside_list = np.concatenate(pool.map(calc_worker, data))

    return is_inside_list.sum()/len(x_points), is_inside_list


def monte_karlo_integrate(x_nodes, y_values, points_amount):
    x_min, x_max = x_nodes.min(), x_nodes.max()
    y_min, y_max = y_values.min(), y_values.max()

    x_points = np.random.sample(points_amount) * (x_max-x_min) + x_min
    y_points = np.random.sample(points_amount) * (y_max-y_min) + y_min

    ratio, is_inside_list = monte_karlo(x_nodes, y_values, x_points, y_points)

    In = (x_max-x_min)*(y_max-y_min)*ratio
    return In, x_points, y_points, is_inside_list
