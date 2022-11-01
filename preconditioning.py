from dolfin import *
from dolfin_adjoint import *

import numpy as np

def preconditioning(func, smoothen=False):
    if not smoothen:
        c = func.copy()
        C = c.function_space()
        dim = c.geometric_dimension()
        BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
        BC.apply(c.vector())
    else:
        C = func.function_space()
        dim = func.geometric_dimension()
        BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
        c = TrialFunction(C)
        psi = TestFunction(C)
        a = inner(grad(c), grad(psi)) * dx
        L = inner(func, psi) * dx
        c = Function(C)
        solve(a == L, c, BC)
    return c
