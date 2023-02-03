from dolfin import *
from dolfin_adjoint import *

import numpy as np

from dgregister.config import hyperparameters
assert len(hyperparameters) > 1

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass



# def preconditioning(func):

#     smoothen = hyperparameters["smoothen"]

#     print("smoothen=", smoothen)


#     if not smoothen:
#         c = func.copy()
#         C = c.function_space()
#         dim = c.geometric_dimension()
#         BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
#         BC.apply(c.vector())
#     else:
#         C = func.function_space()
#         dim = func.geometric_dimension()
#         BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
#         c = TrialFunction(C)
#         psi = TestFunction(C)
#         a = inner(grad(c), grad(psi)) * dx
#         L = inner(func, psi) * dx
#         c = Function(C)
#         solve(a == L, c, BC)
#     return c






class Preconditioning():

    def __init__(self) -> None:
        # self.smoothen = hyperparameters["smoothen"]
        # self.hyperparameters = hyperparameters

        pass

    def __call__(self, func):

        if not hyperparameters["smoothen"]:

            # print_overloaded("applying BC to func in Preconditioning()")
            cc = func.copy()

            # breakpoint()

            # print_overloaded("Debugging: cc ", cc.vector()[:].min(), cc.vector()[:].max(), cc.vector()[:].mean())
            C = cc.function_space()
            dim = cc.geometric_dimension()
            BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
            BC.apply(cc.vector())

            # print_overloaded("Debugging: cc ", cc.vector()[:].min(), cc.vector()[:].max(), cc.vector()[:].mean())

        else:
            C = func.function_space()
            dim = func.geometric_dimension()
            BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
            c = TrialFunction(C)
            psi = TestFunction(C)
            

            if not hasattr(self, "solver"):
                a = inner(grad(c), grad(psi)) * dx
                # a = inner(grad(c), grad(psi)) * dx
                self.A = assemble(a)
                print_overloaded("Assembled A in Preconditioning()")
            
            L = inner(func, psi) * dx
            
            
            tmp = assemble(L)
            BC.apply(tmp)
            
            BC.apply(self.A)
            # solve(a == L, c, BC)

            if not hasattr(self, "solver"):

                if hyperparameters["solver"] == "lu":
                    
                    self.solver = LUSolver()
                    self.solver.set_operator(self.A)

                    print_overloaded("Created LU solver in Preconditioning()")

                elif hyperparameters["solver"] == "krylov":
                    # self.solver = PETScKrylovSolver(method="gmres", preconditioner=self.hyperparameters["preconditioner"])
                    self.solver = PETScKrylovSolver("gmres", hyperparameters["preconditioner"])
                    self.solver.set_operators(self.A, self.A)

                    print_overloaded("Created Krylov solver in Preconditioning()")


            # BC.apply(self.A)
            # x = args[0]
            # b = args[1]

            cc = Function(C)
            self.solver.solve(cc.vector(), tmp)

        return cc


preconditioning = Preconditioning()