from dolfin import *

def transformation(func, diag_matrix):
    if isinstance(func, Function):
        tmp = Function(func.function_space())
        vec = diag_matrix * func.vector()
        tmp.vector()[:] = vec[:]
    else:
        tmp = diag_matrix * func
    return tmp


