from dolfin import *
from dolfin_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

import numpy as np

from preconditioning import preconditioning

backend_preconditioning = preconditioning

class PreconditioningBlock(Block):
    def __init__(self, func, **kwargs):
        super(PreconditioningBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)

    def __str__(self):
        return 'PreconditioningBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        breakpoint()
        c = inputs[0]
        C = c.function_space()
        v = TestFunction(C)
        u = TrialFunction(C)
        mass_form = v * u * dx
        mass_action_form = action(mass_form, interpolate(Constant(1.0), C))
        M_diag = assemble(mass_action_form)
        M_lumped_m05 = assemble(mass_form)
        M_lumped_m05.zero()
        M_diag_m05 = assemble(mass_action_form)
        M_diag_m05.set_local(np.ma.power(M_diag.get_local(), -0.5))
        M_lumped_m05.set_diagonal(M_diag_m05)

        adj_input = adj_inputs[0]
        adj = M_lumped_m05 * adj_input
        adjv = Function(C)
        adjv.vector().set_local(adj)
        return adj

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_preconditioning(inputs[0])

preconditioning = overload_function(preconditioning, PreconditioningBlock)