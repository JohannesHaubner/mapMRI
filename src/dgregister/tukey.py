from fenics import *
from fenics_adjoint import *
from ufl import sign
import numpy as np


# def abs(x):
#     if isinstance(x, np.ndarray):
#         return np.abs(x)
#     else:
#         return x + sign(x) * x

def ReLU(x):
    return (x + abs(x)) / 2


def sigma_estimate(residual, vol, dx):

    # x = ReLU(abs(residual))

    mean = assemble(residual * dx) / vol
    
    x = residual - mean

    # x = ReLU(abs(residual - mean))

    meandev = assemble(x ** 2 * dx) / vol
    
    meandev = sqrt(meandev)
    # 1.486 * 
    return meandev
    
def sigma(residual, vol=None, dx=None):

    if not isinstance(residual, np.ndarray):
        return sigma_estimate(residual=residual, vol=vol, dx=dx)
    else:
        return 1.486 * np.median(np.abs(residual[np.abs(residual)>0] - np.median(residual[np.abs(residual)>0])))

        # return 1.486 * np.median(np.abs(residual[np.abs(residual)>0] - np.median(residual[np.abs(residual)>0])))


    
def sigma2(residual, vol=None, dx=None):

    if not isinstance(residual, np.ndarray):
        return sigma_estimate(residual=residual, vol=vol, dx=dx)
    else:
        return 1.486 * np.median(np.abs(residual - np.median(np.abs(residual))))




def scaled_residual(x, y, dx=None):
    if isinstance(x, np.ndarray):

        s = np.sum(y) / np.sum(x)

    else:
        s = assemble(y*dx) / assemble(x*dx)
        
    return x * np.sqrt(s) - y / np.sqrt(s)

def heavyside(x):

    return (sign(x) + 1) / 2

def tukey(x, c=4.68):

    if isinstance(x, np.ndarray):
        a = c ** 2 / 2 * (1 - (1 - x ** 2 / (c ** 2)) **3 )
        b = c ** 2 / 2
        y = np.where(np.abs(x) <= c, a, b)
    
    else:
        a = c ** 2 / 2 * (1 - (1 - x ** 2 / (c ** 2)) **3 )
        b = c ** 2 / 2
        x = abs(x)
        y = a * heavyside(c-x) + b * heavyside(x - c)

    return y


def tukeyloss(x, y, vol=None, dx=None, c=4):
    r = scaled_residual(x, y, dx=dx)
    factor =  sigma(r, vol, dx=dx)

    if isinstance(x, np.ndarray):
        return tukey(r / factor, c=c)
    else:
        return assemble(tukey(r / factor, c=c) * dx)

def l2loss(x, y, vol=None, dx=None):
    r = scaled_residual(x, y, dx)
    factor =  sigma(r, vol=vol, dx=dx)
    
    if isinstance(x, np.ndarray):
        return (r / factor)**2
    else:
        return assemble((r / factor) ** 2 * dx)

