from dolfin import *
from dolfin_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

from dgregister.preconditioning import preconditioning


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
        
        omega = 0
        epsilon = 1
        
        omega = 0.5
        epsilon = 0.5
        print_overloaded("Using non-default omega=", omega, "epsilon=", epsilon, "in preconditioning_overloaded")

        if not hasattr(self, "solver"):
            
            a = omega * inner(c, psi) * dx + epsilon * inner(grad(c), grad(psi)) * dx
            # a = inner(grad(c), grad(psi)) * dx

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