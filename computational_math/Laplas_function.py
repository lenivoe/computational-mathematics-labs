import numpy as np
from scipy.special import erf

from computational_math.utils import bin_search


def Laplas(x):
    '''
        vectorized Laplas function

        F(x) = 2/sqrt(2*Pi) * integral_(e^(-(t^2)/2) dt)_(t=0..x)

        or
        
        F(x) = erf(x/sqrt(2))
    '''
    return erf(x * (2**0.5 / 2))


__X_MIN = 0
__X_MAX = 5
DIV = 1000
__y_vec = Laplas(np.linspace(__X_MIN, __X_MAX, (__X_MAX-__X_MIN)*DIV+1))

def approx_inv_Laplas(y):
    '''F^(-1)(y), nonvectorized'''
    # так как разбиение равномерное, индекс соответствует x = i / <разбиение>
    i = bin_search(y, __y_vec)
    return i / DIV

def approx_Laplas(x):
    '''Laplas function for x == round(x, 3), nonvectorized'''
    return __y_vec[int(x * 1000)]