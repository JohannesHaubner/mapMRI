from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pytest

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))

# 
from preconditioning_overloaded import preconditioning

from config import hyperparameters

@pytest.mark.parametrize(
    "smoothen", [True, False]
)






def test_preconditioning(smoothen):

    
    mesh = UnitSquareMesh(10, 10)

    hyperparameters["smoothen"] = smoothen

    # hyperparameters["solver"] = "krylov"
    hyperparameters["preconditioner"] = "amg"

    hyperparameters["solver"] = "lu"

    # initialize trafo
    vCG = VectorFunctionSpace(mesh, "CG", 1)

    set_working_tape(Tape())

    # preconditioning = preconditioning(hyperparameters={"smoothen": smoothen, "solver": "lu"})

    # initialize control
    controlfun = interpolate(Constant((1., 1.)), vCG)
    # control = controlfun
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
    for smoothen in [False, True]:
        test_preconditioning(smoothen)
