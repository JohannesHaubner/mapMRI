from dolfin import *
from dolfin_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function
import dolfin.fem as df

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass

from dgregister.preconditioning import preconditioning

from dgregister.config import ALPHA, BETA
from copy import deepcopy

backend_preconditioning = preconditioning




class PreconditioningBlock(Block):
    def __init__(self, func, **kwargs):
        super(PreconditioningBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)
        self.A = None

    def __str__(self):
        return 'PreconditioningBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):

        tmp = adj_inputs[0].copy()
        C = inputs[idx].function_space()
        dim = inputs[idx].geometric_dimension()
        BC = DirichletBC(C, Constant((0.0,) * dim), "on_boundary")
        c = TrialFunction(C)
        psi = TestFunction(C)
        
        alpha = deepcopy(ALPHA)
        beta = deepcopy(BETA)
        
        
        dx = df.form.ufl.dx(C.mesh())

        if alpha != 0 or beta != 1:
            print_overloaded("Using non-default alpha=", alpha, "beta=", beta, "in preconditioning_overloaded")
        else:
            print_overloaded("Using standard alpha, beta", alpha, beta, "in preconditioning_overloaded")
        
        if isinstance(alpha, float):

            alpha = Constant(alpha)
            beta = Constant(beta)

        if not hasattr(self, "solver"):
            
            a = alpha * inner(c, psi) * dx + beta * inner(grad(c), grad(psi)) * dx

            if self.A is None:
                self.A = assemble(a)
            else:
                self.A = assemble(a, tensor=self.A)

            BC.apply(self.A)

            print_overloaded("Assembled A in PreconditioningBlock()")
            
            self.solver = PETScKrylovSolver("gmres", "amg")

            print_overloaded("Created Krylov solver in PreconditioningBlock()")

            self.solver.set_operators(self.A, self.A)
        
        ct = Function(C)
        
        BC.apply(tmp)
        self.solver.solve(ct.vector(), tmp)

        ctest = TestFunction(C)
        tmp = assemble(inner(ctest, ct) * dx)
    
    
        return tmp

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_preconditioning(inputs[0])

preconditioning = overload_function(preconditioning, PreconditioningBlock)