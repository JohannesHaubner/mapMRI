from dolfin import *
from dolfin_adjoint import *

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args, flush=True)
    else:
        pass

from dgregister.config import ALPHA, BETA
from copy import deepcopy



class Preconditioning():

    def __init__(self) -> None:
        self.tmp = None
        self.A = None
        

    def __call__(self, func):

        C = func.function_space()
        dim = func.geometric_dimension()
        BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
        c = TrialFunction(C)
        psi = TestFunction(C)

        alpha = deepcopy(ALPHA)
        beta = deepcopy(BETA)

        if alpha != 0 or beta != 1:
            print_overloaded("Using non-default alpha=", alpha, "beta=", beta, "in preconditioning")

        else:
            print_overloaded("Using standard alpha, beta", alpha, beta, "in preconditioning")

        if isinstance(alpha, float):

            alpha = Constant(alpha)
            beta = Constant(beta)
 
        if not hasattr(self, "solver"):

            a = alpha * (inner(c, psi) * dx) + beta * (inner(grad(c), grad(psi)) * dx)

            if self.A is None:

                self.A = assemble(a)

            else:
                self.A = assemble(a, tensor=self.A)

            BC.apply(self.A)

            self.solver = PETScKrylovSolver("gmres", "amg")
            self.solver.set_operators(self.A, self.A)
            print_overloaded("Created Krylov solver in Preconditioning()")

        L = inner(func, psi) * dx

        if self.tmp is None:
            self.tmp = assemble(L)
        else:
            self.tmp = assemble(L, tensor=self.tmp)
        
        BC.apply(self.tmp)

        cc = Function(C)
        self.solver.solve(cc.vector(), self.tmp)

        return cc


preconditioning = Preconditioning()