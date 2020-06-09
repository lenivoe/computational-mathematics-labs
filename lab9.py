import math
from itertools import takewhile, dropwhile

import matplotlib.pyplot as plt
import numpy as np

import computational_math.integral as cmi


DATA_DIR = 'data/lab9'
IMG_DIR = f'{DATA_DIR}/img'



def main():
    func_tests = {
        'x^4' : (lambda x: x**4, lambda x: (x**5) / 5),
        'sin(x)' : (np.sin, lambda x: -np.cos(x)),
        '-2x*e^(-x^2)' : (lambda x: -2*x*np.exp(-x**2), lambda x: np.exp(-x**2)),
    }

    integration_rules = {
        'формула левых прямоугольников'   : cmi.gen_integral_func(cmi.table_left_Riemann_sum),
        'формула правых прямоугольников'  : cmi.gen_integral_func(cmi.table_right_Riemann_sum),
        'формула трапеций'                : cmi.gen_integral_func(cmi.table_trapezoidal_rule),
        'формула средних прямоугольников' : cmi.midpoint_rule,
        'формула Симпсона'                : cmi.gen_integral_func(cmi.table_Simpsons_rule),
    }


    ifname, ofname = f'{DATA_DIR}/input.txt', f'{DATA_DIR}/output.txt'
    with open(ifname, 'r', encoding='utf-8') as reader, open(ofname, 'w', encoding='utf-8') as writer:
        reader_it = map(str.strip, reader)
        reader_it = filter(lambda s: not s.startswith('#'), reader_it)
        reader_it = takewhile(lambda s: s != '--break', reader_it)

        while True:
            try:
                it = dropwhile(lambda s: s == '', reader_it)
                it = takewhile(lambda s: s != '', it)

                try:
                    method_type = next(it)
                except StopIteration:
                    break
                
                func_name = next(it)
                func, int_func = func_tests[func_name]

                a, b = map(float, next(it).split())
                
                real_int_val = int_func(b) - int_func(a)
                print(f'f(x)={func_name}, [a, b] = [{a}, {b}], точное значение: {real_int_val}', file=writer)


                # с постоянным выбором шага
                if method_type == '--direct':
                    while True:
                        rule_name = next(it)
                        rule = integration_rules[rule_name]
                        h_list = map(float, next(it).split())

                        print(rule_name, file=writer)

                        for h in h_list:
                            int_val = rule(func, a, b, h)
                            real_err = real_int_val - int_val
                            print(f'\th: {h}  ->  интеграл: {int_val},  (I - I_n): {real_err:.2e}', file=writer)
                        print(file=writer)

                # с автоматическим выбором шага
                elif method_type == '--auto':
                    while True:
                        rule_name = next(it)
                        rule = integration_rules[rule_name]
                        err_list = map(float, next(it).split())
                        
                        print(rule_name, file=writer)

                        for err in err_list:
                            int_val, h, amount = cmi.autostep_integrate(func, a, b, err, rule)
                            real_err = real_int_val - int_val
                            print(f'\tпогрешность: {err}  ->  h: {h},  число шагов: {amount},', file=writer)
                            print(f'\t\tинтеграл: {int_val},  (I - I_n): {real_err:.2e}', file=writer)
                        print(file=writer)
                else:
                    raise SyntaxError('unresolved command')
                
            except StopIteration:
                print(file=writer)
                continue


if __name__ == '__main__':
    main()


