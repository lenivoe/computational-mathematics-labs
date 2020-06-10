from itertools import count

import numpy as np



def gen_integral_func(calc_by_table):
    '''
        На основе функции, вычисляющей интеграл по таблице значений,
        генерирует фукнцию, вычисляющую интеграл на интервале с шагом
    '''
    def integrate(func, a, b, h):
        x = np.linspace(a, b, int((b-a)/h) + 1)
        y = func(x)

        assert abs(h - abs(x[1]-x[0])) < 0.000001

        return calc_by_table(y, h)
    
    return integrate


def midpoint_rule(func, a, b, h):
    '''
        Формула средних прямоугольников с генерацией таблицы значений функции.
        Ошибка O(h^2) и в два раза меньше, чем в формуле трапеций.
    '''
    
    x = np.linspace(a+h/2, b-h/2, int((b-a)/h))
    y = func(x)

    assert abs(h - abs(x[1]-x[0])) < 0.000001

    return h * y.sum()


def table_left_Riemann_sum(values :np.ndarray, h :float):
    '''Формула левых прямоугольников. Ошибка O(h)'''
    return h * values[:-1].sum()

def table_right_Riemann_sum(values :np.ndarray, h :float):
    '''Формула правых прямоугольников. Ошибка O(h)'''
    return h * values[1:].sum()

def table_trapezoidal_rule(values :np.ndarray, h :float):
    '''Формула трапеций. Ошибка O(h^2), но в два раза больше, чем у формулы средних прямоугольников'''
    return h * ((values[0]+values[-1])/2 + values[1:-1].sum())

def table_Simpsons_rule(values :np.ndarray, h :float):
    '''Формула Симпсона, количество точек в values должно быть нечетным и >= трех. Ошибка O(h^4).'''

    assert (len(values)-1) % 2 == 0, 'число интервалов должно быть четным'

    return h/3 * (values[0] + values[-1] + 4*values[1:-1:2].sum() + 2*values[2:-2:2].sum())



def autostep_integrate(func, a, b, err, rule = None):
    '''
        Вычисляет результат определенного интеграла от функции func
        - func - подинтегральная функция
        - a, b - пределы интегрирования
        - err - погрешность
        - rule - метод интегрирования, по умолчанию используется формула Симпсона

        return (<значение>, <шаг>, <число итераций>)
    '''

    if rule == None:
        rule = gen_integral_func(table_Simpsons_rule)

    h = (b - a) / 2
    prev_val = rule(func, a, b, h)

    for amount in count(1):
        h /= 2
        cur_val = rule(func, a, b, h)
        if abs(cur_val - prev_val) < err:
            return cur_val, h, amount
        prev_val = cur_val

