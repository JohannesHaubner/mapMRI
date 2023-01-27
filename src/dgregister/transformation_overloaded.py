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

from dolfin_adjoint import *
    
from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

from dgregister.transformation import transformation

backend_transformation = transformation

class TransformationBlock(Block):
    def __init__(self, func, diag_matrix, **kwargs):
        super(TransformationBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)
        self.diag_matrix = diag_matrix

    def __str__(self):
        return 'TransformationBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        tmp = adj_inputs[0].copy()
        tmp = backend_transformation(tmp, self.diag_matrix)
        return tmp

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_transformation(inputs[0], self.diag_matrix)

transformation = overload_function(transformation, TransformationBlock)