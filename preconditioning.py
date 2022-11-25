from fenics import *
from fenics_adjoint import *

import numpy as np


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass



class Preconditioning():

    def __init__(self, hyperparameters) -> None:
        self.smoothen = hyperparameters["smoothen"]
        self.hyperparameters = hyperparameters


    def __call__(self, func):
        
        if not self.smoothen:
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
            

            if not hasattr(self, "solver"):
                a = inner(grad(c), grad(psi)) * dx
                A = assemble(a)
                print_overloaded("Assembled A in Preconditioning()")
            
            L = inner(func, psi) * dx

            tmp = assemble(L)
            c = Function(C)
            
            # solve(a == L, c, BC)

            if not hasattr(self, "solver"):

                if self.hyperparameters["solver"] == "lu":

                    self.solver = LUSolver()
                    self.solver.set_operator(A)

                    print_overloaded("Created LU solver in Preconditioning()")

                elif self.hyperparameters["solver"] == "krylov":
                    self.solver = KrylovSolver(A, "gmres", self.hyperparameters["preconditioner"])
                    self.solver.set_operators(A, A)
                    print_overloaded("Assembled A, using Krylov solver")


            BC.apply(A)
            self.solver.solve(c.vector(), tmp)

        return c


