from dolfin import *
from dolfin_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

import numpy as np

from preconditioning import Preconditioning

preconditioning = Preconditioning()

backend_preconditioning = preconditioning

class PreconditioningBlock(Block):
    def __init__(self, func, smoothen=False, **kwargs):
        super(PreconditioningBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)
        self.smoothen = smoothen

    def __str__(self):
        return 'PreconditioningBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if not self.smoothen:
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

            c = Function(C)
            BC.apply(tmp)

            if not hasattr(self, "solver"):
                a = inner(grad(c), grad(psi)) * dx
                A = assemble(a)
                print("Assembled A in PreconditioningBlock()")
                

                self.solver = LUSolver()
                self.solver.set_operator(A)
                print("Created LU solver in PreconditioningBlock()")
            
            BC.apply(A)
            self.solver.solve(c.vector(), tmp)

            # solve(A, c.vector(), tmp)
            tmp = c.vector()
        return tmp

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_preconditioning(inputs[0], self.smoothen)

preconditioning = overload_function(preconditioning, PreconditioningBlock)
