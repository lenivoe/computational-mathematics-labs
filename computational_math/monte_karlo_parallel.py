import math
import multiprocessing as mp

import numpy as np


def gen_is_inside_data(x_vec, y_vec):
    above = (x_vec - np.concatenate([x_vec[-1:], x_vec[:-1]]))
    under = (y_vec - np.concatenate([y_vec[-1:], y_vec[:-1]]))

    factors = np.empty(len(x_vec))
    factors[:] = np.NaN

    np.divide(above, under, out=factors, where=under!=0)
    return factors


def is_inside(x, y, x_vec, y_vec, factors):
    is_in = False

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


def calc_worker(args):
    x_vec, y_vec, x_nodes, y_values, check_data = args
    inside_amount = 0
    for x, y in zip(x_vec, y_vec):
        if is_inside(x, y, x_nodes, y_values, check_data):
            inside_amount += 1
    return inside_amount


def monte_karlo(x_nodes, y_values, x_points, y_points):
    check_data = gen_is_inside_data(x_nodes, y_values)
    
    cpu = mp.cpu_count()
    job_size = math.ceil(len(x_points)/cpu)
    ab_list = zip(range(0, cpu*job_size, job_size), range(job_size, (cpu+1)*job_size, job_size))

    vecs = ((x_points[a:b], y_points[a:b], x_nodes, y_values, check_data) for a, b in ab_list)

    with mp.Pool(cpu) as pool:
        amount_list = pool.map(calc_worker, vecs)
        inside_amount = sum(amount_list)
    
    return inside_amount/len(x_points)


def monte_karlo_integrate(x_nodes, y_values, points_amount):
    x_min, x_max = x_nodes.min(), x_nodes.max()
    y_min, y_max = y_values.min(), y_values.max()

    x_points = np.random.sample(points_amount) * (x_max-x_min) + x_min
    y_points = np.random.sample(points_amount) * (y_max-y_min) + y_min

    part = monte_karlo(x_nodes, y_values, x_points, y_points)

    In = (x_max-x_min)*(y_max-y_min)*part

    return In, x_points, y_points