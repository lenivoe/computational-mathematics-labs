import numpy as np


def midpoint_rule_for_func(func, a, b, h):
    'формула средних прямоугольников с генерацией таблицы значений функции. Ошибка O(h^2).'
    
    return h * func(np.arange(a, b, h)).sum()


def left_Riemann_sum(values :np.ndarray, h :float):
    'формула левых прямоугольников. Ошибка O(h)'
    return h * values[:-1].sum()

def right_Riemann_sum(values :np.ndarray, h :float):
    'формула правых прямоугольников. Ошибка O(h)'
    return h * values[1:].sum()

def trapezoidal_rule(values :np.ndarray, h :float):
    'формула трапеций. Ошибка O(h^2), но в два раза больше, чем у формулы средних прямоугольников'
    return h * ((values[0]+values[-1])/2 + values[1:-1].sum())


def Simpsons_rule(values :np.ndarray, h :float):
    'Формула Симпсона, количество точек в values должно быть нечетным и >= трех. Ошибка O(h^4).'

    assert len(values) % 2 == 1

    return h/3 * (values[0] + values[-1] + 4*values[1:-1:2].sum() + 2*values[2:-2:2].sum())

