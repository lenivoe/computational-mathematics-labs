import numpy as np

import computational_math.tridiagonal_matrix_algorithm as tma
from computational_math.utils import bin_search


def get_spline_derivative(nodes, b, c, d):
    '''
        Возвращает интерполированную функцию по данным, расчитанным функцией calc_spline_data из таблицы значений функции.
        - nodes - список узлов;
        - a, b, c, d - коэффициенты сплана.
    '''

    def derivative(x):
        i = bin_search(x, nodes)-1
        dx = x - nodes[i+1]
        
        # bi + ci*(x-xi) + di/2*(x-xi)^2
        return b[i] + c[i]*dx + d[i]/2*dx**2

    return derivative


def get_spline(nodes, a, b, c, d):
    '''
        Возвращает интерполированную функцию по данным, расчитанным функцией calc_spline_data из таблицы значений функции.
        - nodes - список узлов;
        - a, b, c, d - коэффициенты сплана.
    '''

    def spline(x):
        i = bin_search(x, nodes)-1
        dx = x - nodes[i+1]
        
        # ai + bi*(x-xi) + ci/2*(x-xi)^2 + di/6*(x-xi)^3
        return a[i] + b[i]*dx + c[i]/2*dx**2 + d[i]/6*dx**3

    return spline


def calc_spline_data(nodes :np.ndarray, values :np.ndarray):
    '''
        Возвращает вектора с коэффициентами a, b, c, d, необходимые для функции сплайна.
        - nodes - список узлов;
        - values - список значений интерполируемой функции в узлах.
    '''

    assert len(nodes) == len(values)

    hx = nodes[1:] - nodes[:-1]
    hy = values[1:] - values[:-1]

    a = tuple(values[1:])

    c = np.array(tma.calc(_gen_tma_data(hx, hy)))
    c[0] = c[-1] = 0 # точные нули на всякий
    d = (c[1:] - c[:-1]) / hx
    
    c = c[1:]

    b = hx*c/2 - hx*hx*d/6 + hy/hx

    return a, b, c, d



def _gen_tma_data(hx, hy):
    assert len(hx) == len(hy)

    yield (0, 1, 0, 0)

    for i in range(len(hy)-1):
        yield (hx[i], 2*(hx[i]+hx[i+1]), hx[i+1], 6*(hy[i+1]/hx[i+1] - hy[i]/hx[i]))
    
    yield (0, 1, 0, 0)


