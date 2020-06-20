import numpy as np


def _calc_k(eq_system, x, y_list, h):
    k1 = np.array([h * f(x, *y_list) for f in eq_system])

    cur_x = x+h/2
    cur_y_list = y_list+k1/2
    k2 = np.array([h * f(cur_x, *cur_y_list) for f in eq_system])

    cur_y_list = y_list+k2/2
    k3 = np.array([h * f(cur_x, *cur_y_list) for f in eq_system])

    cur_x = x+h
    cur_y_list = y_list+k3
    k4 = np.array([h * f(cur_x, *cur_y_list) for f in eq_system])

    return k1, k2, k3, k4


def Runge_Kutta_method(eq_system, y0_list, a, b, h):
    '''возвращает вектор X и матрицу Y, в которой Y[i] соответствует X[i]'''

    x_vec = np.concatenate([np.arange(a, b, h), [b]])
    y_mx = [y0_list, ]

    for x in x_vec[:-1]:
        k1, k2, k3, k4 = _calc_k(eq_system, x, y_mx[-1], h)
        k = (k1 + k2 * 2 + k3 * 2 + k4) / 6
        y_mx.append(y_mx[-1] + k)

    return x_vec, np.array(y_mx)


def Runge_Kutta_with_auto_step(eq_system, y0_list, a, b, err):
    '''возвращает вектор X и матрицу Y, в которой Y[i] соответствует X[i]'''

    H_MIN, E_MIN = (b-a)/10**4,  err/2**5

    def calc_h(eq_system, y_list, a, b, h, err):
        if h > b - a:
            h = b - a

        while True:
            k1, k2, k3, k4 = _calc_k(eq_system, a, y_list, h)
            E = (abs(k1 - k2 - k3 + k4)*(2/3)).max()
            if h < H_MIN or E < err:
                return h, E
            h /= 2

    h = b-a
    cur_a, cur_y_list = a, y0_list
    x_vec, y_mx = [a], [cur_y_list, ]

    while cur_a < b:
        h, cur_err = calc_h(eq_system, cur_y_list, cur_a, b, h, err)
        _, y_list = Runge_Kutta_method(eq_system, y_mx[-1], cur_a, cur_a+h, h)
        cur_a += h

        x_vec.append(cur_a)
        y_mx.append(y_list[-1])

        if cur_err < E_MIN or h < H_MIN:
            h *= 2

    return x_vec, np.array(y_mx)


def global_error_norm(x_list, y_mx, eq_system):
    y_origin_mx = np.array([[f(x) for f in eq_system] for x in x_list])
    return np.abs((y_origin_mx - y_mx)).max()
