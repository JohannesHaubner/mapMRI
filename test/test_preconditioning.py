from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))

from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pytest

from dgregister.preconditioning_overloaded import preconditioning

@pytest.mark.parametrize(
    "smoothen", [True]
)
def test_preconditioning():
    mesh = UnitSquareMesh(10, 10)

    # initialize trafo
    vCG = VectorFunctionSpace(mesh, "CG", 1)

    set_working_tape(Tape())

    # initialize control
    controlfun = interpolate(Constant((1., 1.)), vCG)
    control = preconditioning(controlfun)

    # objective
    J = assemble((control - Constant((1.0, 1.0)))**2*dx)

    c = Control(controlfun)
    Jhat = ReducedFunctional(J, c)

    # first order test
    rate = taylor_test(Jhat, controlfun, controlfun)
    print(rate)
    assert rate > 1.8


if __name__ == "__main__":
    test_preconditioning()


