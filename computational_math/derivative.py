import numpy as np


def calc_first_derivative_values(values :np.ndarray, h :float) -> np.ndarray:
    '''
        Вычисляет значения первой производной для всех заданных значений функции, кроме последнего
        
        Args:
        - values - список значений функции
        - h - шаг между точками, в которых вычеслелны значения функции

        Returns:
        - список значений первой производной для всех значений функции, кроме посленего
    '''

    assert isinstance(values, np.ndarray)

    first = -3*values[0] + 4*values[1] - values[2]
    last = values[-3] - 4*values[-2] + 3*values[-1]
    return np.concatenate([[first], (values[2:]-values[:-2]), [last]]) / (2*h)

def calc_second_derivative_values(values :np.ndarray, h :float) -> np.ndarray:
    '''
        Вычисляет значения второй производной для всех заданных значений функции, кроме первого и последнего
        
        Args:
        - values - список значений функции
        - h - шаг между точками, в которых вычеслелны значения функции

        Returns:
        - список значений второй производной для всех значений функции, кроме первого и последнего
    '''
    
    assert isinstance(values, np.ndarray)

    return (values[:-2] - 2*values[1:-1] + values[2:]) / h**2

def calc_third_derivative_values(values :np.ndarray, h :float) -> np.ndarray:
    '''
        Вычисляет значения третьей производной для всех заданных значений функции,
        кроме первых двух и последних двух
        
        Args:
        - values - список значений функции
        - h - шаг между точками, в которых вычеслелны значения функции

        Returns:
        - список значений третьей производной для всех значений функции,
        кроме первых двух и последних двух
    '''

    assert isinstance(values, np.ndarray)
    
    return (-values[:-4] + 2*values[1:-3] - 2*values[3:-1] + values[4:]) / (2 * h**3)

