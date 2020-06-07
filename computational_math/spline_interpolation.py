import numpy as np

import computational_math.tridiagonal_matrix_algorithm as tma

def getInterpolatedFunc(nodes_x :np.ndarray, nodes_y :np.ndarray):
    '''
        Возвращает интерполированную функцию по сетке из услов и значениям интерполируемой функции в них.
        - nodes_x - список узлов;
        - nodes_y - список значений интерполируемой функции в узлах.
    '''

    assert len(nodes_x) == len(nodes_y)

    hx = nodes_x[1:] - nodes_x[:-1]
    hy = nodes_y[1:] - nodes_y[:-1]

    a = tuple(nodes_y[1:])

    c = np.array(tma.calc(_gen_tma_data(hx, hy)))
    c[0] = c[-1] = 0 # точные нули на всякий
    d = (c[1:] - c[:-1]) / hx
    
    c = c[1:] / 2
    d /= 6

    b = hx*c - hx*hx*d + hy/hx

    def spline(x):
        i = _bin_search(x, nodes_x)-1
        dx = x - nodes_x[i+1]
        
        # a_i + b_i*(x-x_i) + c_i/2*(x-x_i)^2 + d_i/6*(x-x_i)^3
        return a[i] + b[i]*dx + c[i]*dx*dx + d[i]*dx*dx*dx

    return spline

def _gen_tma_data(hx, hy):
    assert len(hx) == len(hy)

    yield (0, 1, 0, 0)

    for i in range(len(hy)-1):
        yield (hx[i], 2*(hx[i]+hx[i+1]), hx[i+1], 6*(hy[i+1]/hx[i+1] - hy[i]/hx[i]))
    
    yield (0, 1, 0, 0)

def _bin_search(x, vec_x):
    lt, rt = 0, len(vec_x)-1
    while lt < rt:
        cur = lt+(rt-lt+1)//2
        if vec_x[cur-1] <= x and x <= vec_x[cur]:
            return cur
        if x < vec_x[cur-1]:
            rt = cur-1
        else:
            lt = cur
    return 0 if x < vec_x[0] else len(vec_x)-1
