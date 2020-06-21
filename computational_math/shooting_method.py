import numpy as np

from computational_math.runge_kutta import Runge_Kutta_with_auto_step
from computational_math.root_finding import secant_method


def _gen_func_get_approx_last_values(system, a, b, err, get_first_values):
    ''' генерирует функцию, возвращающую приблизительные значения в последней точке функции '''
    def get_approx_last_values(a_param):
        a_values = np.array(get_first_values(a_param))
        _, y_mx = Runge_Kutta_with_auto_step(system, a_values, a, b, err)
        return y_mx[-1]

    return get_approx_last_values


def _gen_func_check_last_values(get_approx_last_values, last_boundary_condition):
    ''' возвращает уравнение для правого граниченого условия от приближения для левого граниченого условия '''
    def check_last_values(a_param):
        return last_boundary_condition(*get_approx_last_values(a_param))

    return check_last_values


def shooting_method(
        eq_system,
        a, b,
        err,
        param0, param1,
        get_first_values,
        second_boundary_condition
):
    '''
        Находит левые первые значения и решает задачу Коши методом Рунге-Кутта.
        - eq_system - система дифференциальных уравнений,
        - a, b - точки граничных условий,
        - err - точность метода,
        - param0, param1 - приближения к y(a) или u(a),
        - get_first_values - по одному из значений y(a), u(a) возвращает оба,
        - second_boundary_condition - левая часть граничного условия вида fi2(y(b), u(b)) = 0.
    '''

    get_approx_last_values = _gen_func_get_approx_last_values(eq_system, a, b, err, get_first_values)
    check_last_values = _gen_func_check_last_values(get_approx_last_values, second_boundary_condition)

    correct_a_param = secant_method(check_last_values, param0, param1, err)
    y_a, u_a = get_first_values(correct_a_param)
    a_values = np.array([y_a, u_a])
    return Runge_Kutta_with_auto_step(eq_system, a_values, a, b, err)
