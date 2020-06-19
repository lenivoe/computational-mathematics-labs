import numpy as np

def calc_k(equ_system, x, y, h) :
    k1 = np.array([h * f(x, *y) for f in equ_system])

    tmp_y, tmp_x = y+k1/2, x+h/2
    k2 = np.array([h * f(tmp_x, *tmp_y) for f in equ_system])

    tmp_y = y+k2/2
    k3 = np.array([h * f(tmp_x, *tmp_y) for f in equ_system])

    tmp_y, tmp_x = y+k3, x+h
    k4 = np.array([h * f(tmp_x, *tmp_y) for f in equ_system])

    return k1, k2, k3, k4

def runge_kutta_method(equ_system, y0:np, a, b, h):
    x = np.concatenate([np.arange(a, b, h), [b]])
    cur_y = y0.copy()
    result_y = [cur_y, ]

    for cur_x in x[:-1] :
        k1, k2, k3, k4 = calc_k(equ_system, cur_x, cur_y, h)
        k = (k1 + k2 * 2 + k3 * 2 + k4) / 6
        cur_y = cur_y + k
        result_y.append(cur_y)
    
    return x, result_y

def RK_auto_step(equ_system, y0:np, a, b, e) :
    def _calc_h(equ_system, b, cur_a, cur_y, e, h) :
        if cur_a + h > b : h = b - cur_a
        while True:
            k1, k2, k3, k4 = calc_k(equ_system, cur_a, cur_y, h)
            err = ((k1 - k2 - k3 + k4)*2/3).max()
            if err > e :
                h /= 2
            elif err <= e or h < H_MIN :
                return h, err

    H_MIN, E_MIN = (b-a)/10000,  e/2**5
    h = b-a
    cur_a, cur_y = a, y0.copy()
    result_x, result_y = [a], [cur_y, ]

    while cur_a < b :
        h, err = _calc_h(equ_system, b, cur_a, cur_y, e, h)
        _, new_y = runge_kutta_method(equ_system, result_y[-1], cur_a, cur_a+h, h)
        cur_a += h
        result_x.append(cur_a)
        result_y.append(new_y[-1])
        if err < E_MIN or h < H_MIN : h *= 2
    
    return result_x, result_y

def glob_err_norm(x_list, y_ll:np, system) :
    exact_y = np.array([[f(x) for f in system] for x in x_list], float)
    return np.abs((exact_y - y_ll)).max()









# ЛАБА САШИ СКОРРЕКТИРОВАННАЯ СЛЕГКА, ЧТОБЫ ЗАПУСКАТЬСЯ С ФУНКЦИЯМИ ДЛЯ ОЛЕГА
import math
import matplotlib.pyplot as plt
from datetime import datetime
def to_drow(x_list, y_all, a_b, exact_f_s, f_title) :
    fig, ax = plt.subplots()
    ax.set_title(f_title)

    for i in range(len(y_all[0])) :
        ax.plot(x_list, tuple(y[i] for y in y_all), linewidth = 3)
    
    ex_x_list = np.concatenate([np.arange(*a_b, 0.0001), [a_b[1]]])
    for f in exact_f_s :
        ax.plot(ex_x_list, tuple(map(f, ex_x_list)), label='reference')
    now = datetime.now()
    plt.show()
    #fig.savefig('plot/'+str(now.minute).zfill(2) + str(now.microsecond) + '.png', bbox_inches='tight')


def global_err_norm(x_list, all_y, ref_system) :
    all_ref_y = []
    for x in x_list :
        tmp = [f(x) for f in ref_system]
        all_ref_y.append(tmp)
    
    maxs = []
    for y_l, ref_y_l in zip(all_y, all_ref_y) :
        maxs.append(max(abs(y_l[i] - ref_y_l[i]) for i in range(len(y_l))))
    return max(maxs)

def main() :
    f_system_list = (
        [lambda x, y: 3 * x**2 + (y - x**3)],
        [lambda x, y: 3 * x**2 - 10 * (y - x**3)] #,
        # [lambda x, y1, y2: -2*y1 + 4*y2, lambda x, y1, y2: -y1 + 3*y2]

    )
    syst_title_list = (
        'y\' = 3 x + y - x**3',
        'y\' = 3 x - 10 (y - x**3)',
        'y1\'= -2*y1 + 4*y2 \ny2\'= -y1 + 3y2'
    )
    reference_f_s_list = (
        [lambda x: x**3],
        [lambda x: x**3],
        [lambda x: 4 * math.exp(-x) - math.exp(2 * x),
            lambda x: math.exp(-x) - math.exp(2 * x)]
    )
    a_b = (0, 0.5)

    h_list = (0.01, )
    #e_list = (0.1, 0.01, 0.001)
    e_list = (0.0001, )
    text = ''

    for system, ref_syst, syst_title in zip(f_system_list, reference_f_s_list, syst_title_list) :
        a, b = a_b
        y0_list = np.array(tuple(f(a) for f in ref_syst), float)

        # ПОСТОЯННЫЙ ШАГ
        text += syst_title + f'\tна интервале [{a}, {b}]\n' + 'Постоянный шаг: \n'
        for h in h_list :
            # Вычисления
            x_list, all_y = runge_kutta_method(system, y0_list, *(a_b), h)
            
            # Отрисовка
            title = syst_title + '\n' + f'Шаг = {h}'
            to_drow(x_list, all_y, a_b, ref_syst, title)

            # Вычисление нормы глобальной погрешности
            glob_e = glob_err_norm(x_list, all_y, ref_syst)
            # Вывод нормы г.п.
            text += f'\tШаг = {h}'.ljust(15) + f'Норма глоб.погр-ти = {glob_e : 0.10f}' + '\n'
        text += '\n'

        # АВТОМАТИЧЕСКИЙ ВЫБОР ШАГА
        text += syst_title + f'\tна интервале [{a}, {b}]\n' + 'Автоматический выбор шага: \n'
        for e in e_list:
            # Вычисления
            x_list, all_y = RK_auto_step(system, y0_list, a, b, e)

            # Отрисовка
            title = syst_title + '\n' + f'e = {e}'
            to_drow(x_list, all_y, a_b, ref_syst, title)

            # Вычисление нормы глобальной погрешности
            glob_e = glob_err_norm(x_list, all_y, ref_syst)
            # Вывод нормы г.п.
            text += f'\te = {e}'.ljust(15) + f'Норма глоб.погр-ти = {glob_e : 0.8f}' + '\n'

        text += '\n\n'

    with open('output13.txt', 'a', encoding='utf8') as file :
        file.write(text)

    print(text)

if __name__ == "__main__":
    main()
    # syst_title = 'u\' + 30u = 0'
    # system = [lambda x, y: -30 * y]
    # a, b = 0, 1
    # y0_list = [1]
    # h_list = [1/10, 1/11]

    # text = syst_title + f'\tна интервале [{a}, {b}]\n' + 'Постоянный шаг: \n'
    # for h in h_list :
    #     x_list, all_y = Runge_Kutta_methods(system, y0_list, (a,b), h)
        
    #     title = syst_title + '\n' + f'Шаг = {h}'
    #     to_drow(x_list, all_y, (a,b), [], title)

    # text += '\n'
    
    
    # with open('output13.txt', 'w', encoding='utf8') as file :
    #     file.write(text)

    # print(text)