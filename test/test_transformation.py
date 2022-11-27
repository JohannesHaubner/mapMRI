from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pytest

from transformation_overloaded import transformation

@pytest.mark.parametrize(
    "smoothen", [True, False]
)
def test_transformation(smoothen):
    mesh = UnitSquareMesh(10, 10)

    # initialize trafo
    vCG = VectorFunctionSpace(mesh, "CG", 1)
    s1 = TrialFunction(vCG)
    s2 = TestFunction(vCG)
    form = inner(s1, s2) * dx
    mass_action_form = action(form, Constant((1., 1.)))
    M_lumped = assemble(form)
    M_lumped_inv = assemble(form)
    M_lumped.zero()
    M_lumped_inv.zero()
    diag = assemble(mass_action_form)
    diag[:] = np.sqrt(diag[:])
    diaginv = assemble(mass_action_form)
    diaginv[:] = 1.0/np.sqrt(diag[:])
    M_lumped.set_diagonal(diag)
    M_lumped_inv.set_diagonal(diaginv)

    set_working_tape(Tape())

    # initialize control
    controlfun = interpolate(Constant((1., 1.)), vCG)
    control = transformation(controlfun, M_lumped)

    # objective
    J = assemble(control**2*dx)

    c = Control(controlfun)
    Jhat = ReducedFunctional(J, c)

    # first order test
    assert taylor_test(Jhat, controlfun, controlfun) > 1.8



