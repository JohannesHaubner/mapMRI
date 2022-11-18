from dolfin import *
from dolfin_adjoint import *

import numpy as np


class Preconditioning():

    def __init__(self) -> None:
        pass


    def __call__(self, func, smoothen=False):
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
            

            if not hasattr(self, "solver"):
                a = inner(grad(c), grad(psi)) * dx
                A = assemble(a)
                print("Assembled A in Preconditioning()")
            
            L = inner(func, psi) * dx

            tmp = assemble(L)
            c = Function(C)
            
            # solve(a == L, c, BC)

            if not hasattr(self, "solver"):

                

                self.solver = LUSolver()
                self.solver.set_operator(A)
                print("Created LU solver in Preconditioning()")
            
            BC.apply(A)
            self.solver.solve(c.vector(), tmp)

        return c


