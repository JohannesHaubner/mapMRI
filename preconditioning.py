from dolfin import *
from dolfin_adjoint import *

import numpy as np

def preconditioning(func):
    c = func.copy()
    C = c.function_space()
    dim = c.geometric_dimension()
    BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
    BC.apply(c.vector())
    return c
