from dolfin import *
from dolfin_adjoint import *

import numpy as np

def preconditioning(c):
    C = c.function_space()
    # lumped mass matrix for IPOPT
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
    #precond
    cp_v = M_lumped_m05 * c.vector()
    cp = Function(C)
    cp.vector().set_local(cp_v)
    return cp
