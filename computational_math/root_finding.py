import numpy as np


def sign(x):
    return x and (-1, 1)[int(x > 0)]


def chord(p, q, func):
    return q - func(q)*(p-q)/(func(p)-func(q))


def Newton(t, func, dfunc):
    return t - func(t) / dfunc(t)


def combined(a, b, func, dfunc, d2func):
    if dfunc(a) * d2func(a) >= 0:
        a = chord(a, b, func)
        b = Newton(b, func, dfunc)
    else:
        b = chord(b, a, func)
        a = Newton(a, func, dfunc)

    return a, b


def combined_method(a, b, func, dfunc, d2func, err):
    LIMIT = 10**3

    # при делении на ноль или выходе из области определения функции
    # отлавливается исключение и считается, что решение не найдено
    with np.errstate(all='raise'):
        try:
            for i in range(LIMIT):
                a, b = combined(a, b, func, dfunc, d2func)
                if abs(a-b) < err:
                    return a+(b-a)/2, i
        except FloatingPointError:
            return np.NaN, np.Inf

        return a+(b-a)/2, np.Inf


def find_roots(func, dfunc, d2func, border_points, err):
    '''Находит корни уравнения на указанных отрезках.'''
    roots = []

    for a, b in zip(border_points[:-1], border_points[1:]):
        s = sign(func(a)) * sign(func(b))
        if s == 0:
            if abs(func(a)) < err:
                roots.append((a, 0, a, b))
        elif s < 0:
            x, iters_amount = combined_method(a, b, func, dfunc, d2func, err)
            roots.append((x, iters_amount, a, b))

    if abs(func(b)) < err:
        roots.append((b, 0, a, b))

    return roots


def secant_method(func, prev_x, x, err, need_iters_amount=False):
    LIMIT = 10**3

    # при делении на ноль или выходе из области определения функции
    # отлавливается исключение и считается, что решение не найдено
    with np.errstate(all='raise'):
        try:
            for iters_amount in range(LIMIT):
                next_x = chord(prev_x, x, func)
                if abs(next_x - x) < err:
                    break
                x, prev_x = next_x, x
            else:
                iters_amount = np.Inf

            result_x = x + (next_x - x)/2

        except FloatingPointError:
            result_x, iters_amount = np.NaN, np.Inf

        return (result_x, iters_amount) if need_iters_amount else result_x
