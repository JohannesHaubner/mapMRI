from dolfin import *
from dolfin_adjoint import *

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args, flush=True)
    else:
        pass

def preconditioning(func):

    C = func.function_space()
    dim = func.geometric_dimension()
    BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
    c = TrialFunction(C)
    psi = TestFunction(C)
    
    a = inner(grad(c), grad(psi)) * dx

    A = assemble(a)

    BC.apply(A)

    L = inner(func, psi) * dx


    tmp = assemble(L)
    
    BC.apply(tmp)

    cc = Function(C)
    solve(A, cc.vector(), tmp)

    return cc


preconditioning