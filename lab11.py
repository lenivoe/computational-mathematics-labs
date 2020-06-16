from itertools import takewhile, dropwhile, accumulate
import os

import matplotlib.pyplot as plt
import numpy as np

from computational_math.utils import vectorize, scale_img
from computational_math.root_finding import find_roots


DATA_DIR = 'data/lab11'
IMG_DIR = f'{DATA_DIR}/img'



def plot(func, a, b, roots, need_exclude_correct=True, *, title=''):
    has_errors = any(not (lt <= x <= rt) for x, _, lt, rt in roots)
    if need_exclude_correct and has_errors:
        return True
    

    fig, ax = plt.subplots()

    x = np.linspace(a, b, int((b-a)*150))
    ax.plot(x, np.zeros(len(x)), color='0.7')
    ax.plot(x, func(x))

    if len(roots) > 0:
        kwargs = {
            'linestyle':'',
            'marker':'|',
            'markersize':20,
            'markeredgewidth':2,
            'color':'magenta',
        }

        for _, _, c, d in roots:
            sub_x = x[np.logical_and(c<=x, x<=d)]
            ax.plot(sub_x, func(sub_x), linestyle='dashed', linewidth=2, color='cyan')
            ax.plot([c,d], func(np.array([c,d])), **kwargs)

        roots = np.array([x for x,*_ in roots])
        ax.plot(roots, func(roots), linestyle='', marker='.', color='black')
    
    plt.title(title)
    plt.savefig(f'{IMG_DIR}/{len(os.listdir(IMG_DIR))+1:02}.png', bbox_inches='tight', pad_inches=0)
    #plt.show()
    plt.close(fig)
    
    return has_errors



def main():
    equation_tests = {
        # корни: 0, 3 (*2), 5
        'x^4 - 11*x^3 + 39*x^2 - 45*x = 0' : (
            lambda x: x**4 - 11*x**3 + 39*x**2 - 45*x,
            lambda x: 4*x**3 - 33*x**2 + 78*x - 45,
            lambda x: 12*x**2 - 66*x + 78,
        ),

        # корни: -1, 0.5 + k (k=0,1,-1,2,-2,...)
        'x*cos(pi*x) = 0' : (
            lambda x: x*np.cos(np.pi*x),
            lambda x: np.cos(np.pi*x) - np.pi*x*np.sin(np.pi*x),
            lambda x: -2*np.pi*np.sin(np.pi*x) - (np.pi**2)*x*np.cos(np.pi*x),
        ),

        # корни: 1, 2+2k (k=0,1,2,3,...)
        'ln^2(x)*sin(pi*x/2) = 0' : (
            lambda x: np.log(x)**2 * np.sin(np.pi/2*x),
            lambda x: 2*np.log(x)*np.sin(np.pi/2*x)/x + np.pi/2*np.log(x)**2 * np.cos(np.pi/2*x),
            lambda x: -(((np.pi*x*np.log(x))**2 + 8*np.log(x) - 8)*np.sin(np.pi/2*x) 
                            - 8*np.pi*x*np.log(x)*np.cos(np.pi/2*x))/(4*x**2),
        ),

        'x^3 - 8*x^2 + 12*x = 0' : (
            lambda x: x**3 - 8*x**2 + 12*x,
            lambda x: 3*x**2 - 16*x + 12,
            lambda x: 6*x - 16,
        ),
    }

    for fname in os.listdir(IMG_DIR):
        os.remove(f'{IMG_DIR}/{fname}')
    
    def fmt_info(root_info, func):
        x, iters_amount, a, b = root_info
        s = f'{f"[{a:.3g}, {b:.2g}]":16}'
        if not np.isnan(x):
            s += f'   f(x)={func(x):<8.2g}   число шагов: {iters_amount:g}   x={x:<8.2g}'
            if not (a <= x <= b):
                s += ' (некорректно)'
        if np.isinf(iters_amount):
            s = f'{s:62} (не сошлось)'
        return s

    ifname, ofname = f'{DATA_DIR}/input.txt', f'{DATA_DIR}/output.txt'
    with open(ifname, 'r', encoding='utf-8') as reader, open(ofname, 'w', encoding='utf-8') as writer:
        line_it = map(str.strip, reader)
        line_it = filter(lambda s: s and not s.startswith('#'), line_it)
        line_it = takewhile(lambda s: s != '--break', line_it)


        for equation_name, ab_str, err_str, h_str in zip(*[line_it]*4):
            a, b = map(float, ab_str.split())
            err = float(err_str)
            h = float(h_str)

            func, dfunc, d2func = equation_tests[equation_name]
            points = np.concatenate([np.arange(a, b, h), [b]])
            roots_info = find_roots(func, dfunc, d2func, points, err)

            title = f'{equation_name}, [{a:g}, {b:g}], h={h:g}, err={err:.1e}'
            print(title, file=writer)
            print(*map(lambda info: fmt_info(info, func), roots_info), '', sep='\n', file=writer)

            plot(func, a, b, roots_info, False, title=title)

    for fname in os.listdir(f'{IMG_DIR}'):
        scale_img(f'{IMG_DIR}/{fname}', 0.8)

if __name__ == '__main__':
    main()

