from fenics import *
from fenics_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

from normalise import normalise


backend_normalise = normalise


class NormaliseBlock(Block):
    def __init__(self, func, **kwargs):
        super(NormaliseBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)

    def __str__(self):
        return 'NormaliseBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0].copy(deepcopy=True)
        print("eval adj component")
        breakpoint()
        #return adj_input
        #return backend_preconditioning(adj_input)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        print("hello recompute")
        return backend_normalise(inputs[0])


normalise = overload_function(normalise, NormaliseBlock)
