import numpy as np
from PIL import Image


def vectorize(func):
    '''Векторизирует функцию одного аргумента'''
    
    return lambda vec: np.fromiter(map(func, vec), float, len(vec))

def frange(begin, end, step):
    return np.linspace(begin, end, int((end-begin)/step)+1)

def scale_img(img_name :str, scale :float):
    img = Image.open(img_name)
    img = img.resize((np.array(img.size)*scale).astype(int), Image.ANTIALIAS)
    img.save(img_name) 


def power(x:float, p_up:int, p_down:int):
    assert x >= 0 or (x < 0 and p_down % 2 == 1), 'wrong: negative x with even p_down'

    sign = (1 if x >= 0 or p_up % 2 == 0 else -1)
    return sign * abs(x)**(p_up/p_down)

def bin_search(x, vec_x):
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
