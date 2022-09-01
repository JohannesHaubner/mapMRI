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
        adj_input = adj_inputs[0]
        return backend_preconditioning(adj_input)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_preconditioning(inputs[0])

preconditioning = overload_function(preconditioning, PreconditioningBlock)