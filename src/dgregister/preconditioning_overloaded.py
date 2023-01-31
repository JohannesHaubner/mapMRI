from dolfin import *

# import dgregister.config as config
# def print_overloaded(*args):
#     if MPI.rank(MPI.comm_world) == 0:
#         # set_log_level(PROGRESS)
#         print(*args)
#     else:
#         pass
# # if ocd:

# if "optimize" in config.hyperparameters.keys() and (not config.hyperparameters["optimize"]):
#     print_overloaded("Not importing dolfin-adjoint")
# else:
#     print_overloaded("Importing dolfin-adjoint")
#     from dolfin_adjoint import *

from dolfin_adjoint import *

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args, flush=True)
    else:
        pass

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function


import numpy as np

from dgregister.preconditioning import preconditioning

from dgregister.config import hyperparameters


assert len(hyperparameters) > 1



backend_preconditioning = preconditioning




class PreconditioningBlock(Block):
    
    def __init__(self, func, **kwargs):
        super(PreconditioningBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)
        # self.smoothen = smoothen

        C = func.function_space()
        dim = func.geometric_dimension()
        
        self.C = C
        self.dim = dim
            # if not hasattr(self, "BC"):

        # self.smoothen = hyperparameters["smoothen"]
        self.BC = DirichletBC(C, Constant((0.0,) * dim), "on_boundary")
        self.c = TrialFunction(C)
        self.psi = TestFunction(C)
        self.ct = Function(C)
        self.ctest = TestFunction(C)
        self.matrix = assemble(inner(self.c, self.psi) * dx)
     
        #     if not hasattr(self, "solver"):
        a = inner(grad(self.c), grad(self.psi)) * dx
        self.A = assemble(a)
        self.BC.apply(self.A)

        print_overloaded("Assembled A in PreconditioningBlock() with ", func)
        
        # if True:
        if hyperparameters["solver"] == "lu":

            self.solver = LUSolver()
            self.solver.set_operator(self.A)
            print_overloaded("Created LU solver in PreconditioningBlock()")

        elif hyperparameters["solver"] == "krylov":
            self.solver = PETScKrylovSolver("gmres", hyperparameters["preconditioner"])
            # 
            # print_overloaded("type of A", type(self.A), self.A)
            # print_overloaded("type of self.solver", type(self.solver))
            print_overloaded("Created Krylov solver in PreconditioningBlock() with ", func)

            self.solver.set_operators(self.A, self.A)
    


    def __str__(self):
        return 'PreconditioningBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        
        # print("self.smoothen=", self.smoothen)
        
        if not hyperparameters["smoothen"]:

            C = inputs[idx].function_space()
            dim = Function(C).geometric_dimension()
            BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
            tmp = adj_inputs[0].copy()
            BC.apply(tmp)
        else:
            tmp = adj_inputs[0]
            # tmp = adj_inputs[0].copy()
            
            
            # a = inner(grad(c), grad(psi)) * dx
            # A = assemble(a)
            
            
            self.BC.apply(tmp)
            
            print_overloaded("Solving in preconditioning_overloaded, tmp=", tmp)
            self.solver.solve(self.ct.vector(), tmp)
            # solve(self.A, ct.vector(), tmp)
            
            # tmp = assemble(inner(self.ctest, self.ct) * dx)
            tmp = self.matrix * self.ct.vector()
        
        return tmp

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_preconditioning(inputs[0])

preconditioning = overload_function(preconditioning, PreconditioningBlock)

#preconditioning = lambda x: x