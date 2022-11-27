from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pytest

from preconditioning_overloaded import preconditioning

@pytest.mark.parametrize(
    "smoothen", [True, False]
)
def test_preconditioning(smoothen):
    mesh = UnitSquareMesh(10, 10)

    # initialize trafo
    vCG = VectorFunctionSpace(mesh, "CG", 1)

    set_working_tape(Tape())

    # initialize control
    controlfun = interpolate(Constant((1., 1.)), vCG)
    control = preconditioning(controlfun)

    # objective
    J = assemble(control**2*dx)

    c = Control(controlfun)
    Jhat = ReducedFunctional(J, c)

    # first order test
    rate = taylor_test(Jhat, controlfun, controlfun)
    print(rate)
    assert rate > 1.8


if __name__ == "__main__":
    test_preconditioning(True)


