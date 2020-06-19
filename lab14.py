from computational_math.runge_kutta import Runge_Kutta_with_auto_step
import numpy as np


# метод написан для решения системы из двух уравнений,
# причем первое уравнение (v) имеет начальное условие (решение в точке a, v(a) или v0),
# а второе (u) не имеет, но имеет решение на правой границе интервала (u(b))
# l_r -- интервал, внутри которого (как ожидается) лежит u(a)
# a_b -- интервал для Коши
def shooting_method(eq_system, a, b, l, r, va, ub, h, e):
    segment = find_segment(eq_system, a, b, l, r, va, ub, h, e)
    x_vec, y_vec = find_ua(eq_system, a, b, va, ub, h, e, segment)
    return x_vec, y_vec


def find_segment(eq_system, a, b, l, r, va, ub, h, e):

    def calc_ub(u0):
        _, result = Runge_Kutta_with_auto_step(eq_system, np.array((va, u0)), a, b, e)
        return result[-1][1]

    vector = np.array([np.arange(l, r, h)])

    right_value = calc_ub(l)
    left_value = None
    for i in range(1, len(vector)):
        left_value = right_value
        right_value = calc_ub(vector[i])

        if (ub - left_value) >= 0 and (right_value - ub) > 0:
            return (vector[i-1], vector[i])
    return None


def find_ua(eq_system, a, b, va, ub, h, e, segment):
    lt, rt = segment
    while True:
        mdl = (lt+rt)/2
        x_vector, y_vector = Runge_Kutta_with_auto_step(eq_system, np.array((va, mdl)), a, b, e)
        cur_ub = y_vector[-1][1]
        if abs(cur_ub - ub) < e:
            break
        elif cur_ub > ub:
            rt = mdl
        else:
            lt = mdl

    return x_vector, y_vector
