from fenics import *
from fenics_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

import numpy as np

from preconditioning import Preconditioning


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

class Overloaded_Preconditioning():

    def __init__(self, hyperparameters) -> None:
        
        # self.smoothen = hyperparameters["smoothen"]
        # self.hyperparameters = hyperparameters

        smoothen = hyperparameters["smoothen"]

        self.forward_preconditioning = Preconditioning(hyperparameters)

        backend_preconditioning = self.forward_preconditioning

        class PreconditioningBlock(Block):
            def __init__(self, func, **kwargs):
                super(PreconditioningBlock, self).__init__()
                self.kwargs = kwargs
                self.add_dependency(func)

            def __str__(self):
                return 'PreconditioningBlock'

            def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
                if not smoothen:
                    C = inputs[idx].function_space()
                    dim = Function(C).geometric_dimension()
                    BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
                    tmp = adj_inputs[0].copy()
                    BC.apply(tmp)
                
                else:
                    tmp = adj_inputs[0].copy()
                    C = inputs[idx].function_space()
                    dim = inputs[idx].geometric_dimension()
                    BC = DirichletBC(C, Constant((0.0,) * dim), "on_boundary")
                    c = TrialFunction(C)
                    psi = TestFunction(C)

                    # tmp = adj_inputs[0].copy()
                    # C = inputs[idx].function_space()
                    # dim = inputs[idx].geometric_dimension()
                    # BC = DirichletBC(C, Constant((0.0,) * dim), "on_boundary")
                    # c = TrialFunction(C)
                    # psi = TestFunction(C)
                    # a = inner(grad(c), grad(psi)) * dx
                    # A = assemble(a)

                    # BZ: this was not present in the "working" implementation
                    # it was after assembly !
                    # c = Function(C)
                    
                    
                    BC.apply(tmp)

                    if not hasattr(self, "solver"):
                        a = inner(grad(c), grad(psi)) * dx
                        self.A = assemble(a)
                        print_overloaded("Assembled A in PreconditioningBlock()")
                        
                        if hyperparameters["solver"] == "lu":

                            self.solver = LUSolver()
                            self.solver.set_operator(self.A)
                            print_overloaded("Created LU solver in PreconditioningBlock()")

                        elif hyperparameters["solver"] == "krylov":
                            self.solver = PETScKrylovSolver("gmres", hyperparameters["preconditioner"])
                            # 
                            print_overloaded("type of A", type(self.A), self.A)
                            print_overloaded("type of self.solver", type(self.solver))
                            print_overloaded("Created Krylov solver in PreconditioningBlock()")

                            self.solver.set_operators(self.A, self.A)
                        
                    c = Function(C)
                    BC.apply(self.A)
                    self.solver.solve(c.vector(), tmp)

                    # solve(A, c.vector(), tmp)
                    tmp = c.vector()
                return tmp

            def recompute_component(self, inputs, block_variable, idx, prepared):
                return backend_preconditioning(inputs[0]) # , self.smoothen)

        self.fun = overload_function(self.forward_preconditioning, PreconditioningBlock)
    
    def __call__(self, *args):
        # preconditioning = overload_function(forward_preconditioning, PreconditioningBlock)
        return self.fun(*args)

