import numpy as np
from PIL import Image


def vectorize(func):
    '''Векторизирует функцию одного аргумента'''
    
    return lambda vec: np.fromiter(map(func, vec), float, len(vec))


def scale_img(img_name :str, scale :float):
    img = Image.open(img_name)
    img = img.resize((np.array(img.size)*scale).astype(int), Image.ANTIALIAS)
    img.save(img_name) 


def power(x:float, p_up:int, p_down:int):
    assert x >= 0 or (x < 0 and p_down % 2 == 1), 'wrong: negative x with even p_down'

    sign = (1 if x >= 0 or p_up % 2 == 0 else -1)
    return sign * abs(x)**(p_up/p_down)
