from fenics import *
from fenics_adjoint import *


def normalise(func):
    print("preconditionin")
    c = func.copy(deepcopy=True)
    C = c.function_space()
    dim = c.geometric_dimension()
    BC=DirichletBC(C, Constant(0.0), "on_boundary")
    #BC.apply(c.vector())
    return c.copy(deepcopy=True)

