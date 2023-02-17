from fenics import *
from fenics_adjoint import *
from ufl import sign
import numpy as np


def ReLU(x):
    return (x + abs(x)) / 2


def heavyside(x):

    if isinstance(x, np.ndarray) or isinstance(x, float):
        return (np.sign(x) + 1) / 2
    else:
        return (sign(x) + 1) / 2


def huber_direct(x, delta):

    return np.where(np.abs(x) < delta, x**2/2, delta*(np.abs(x) - delta / 2))

def huber(x, delta):

    a = x ** 2 / 2
    b = delta * (abs(x) - delta / 2)
    x = abs(x)
    y = a * heavyside(delta - x) + b * heavyside(x - delta)

    return y


if __name__ == "__main__":

    x_vals = np.linspace(0, 2, 11)

    delta = 1    

    for x in x_vals:
        print(x, huber(np.array(x), delta), huber_direct(x, delta))# , ",", hub(x))
        assert np.allclose(huber(np.array(x), delta), huber_direct(x, delta))