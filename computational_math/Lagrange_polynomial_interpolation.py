import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import math

def get_Lagrange_polynomial(nodes_x :np.ndarray, nodes_y :np.ndarray):
    '''
        Возвращает интерполированную функцию по сетке из услов и значениям интерполируемой функции в них.
        - nodes_x - список узлов;
        - nodes_y - список значений интерполируемой функции в узлах.
    '''
    assert len(nodes_x) == len(nodes_y)

    size = len(nodes_x)

    # omega(x) = ((x - x_0)*...*(x - x_n)) / (x - x_k)
    def omega(k, x):
        return (x - nodes_x[np.arange(size) != k]).prod()

    # [omega(x)_0, omega(x)_1, ... , omega(x)_(n-1)]
    def omega_vec(x):
        return np.fromiter(map(lambda k: omega(k, x), range(size)), dtype=float)

    vec_d_omega = np.fromiter(map(lambda v: omega(*v), enumerate(nodes_x)), dtype=float)
    vec = nodes_y / vec_d_omega # значения f(x) / omega'(x)

    return lambda x: (omega_vec(x) * vec).sum()
