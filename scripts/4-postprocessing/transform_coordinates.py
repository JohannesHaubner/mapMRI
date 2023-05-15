"""
Create a FEniCS function that maps coordinates by transport via velocity field.
The initial condition is the vector field given by (x, y, z). 
"""

import os
import argparse
import json
from fenics import *
from fenics_adjoint import *
import pathlib



def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass

parser = argparse.ArgumentParser()

parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--outputfilename", type=str, default="all.hdf")
parserargs = vars(parser.parse_args())


deformation_hyperparameters = json.load(open(parserargs["folder"] + "hyperparameters.json"))

import dgregister.config as config

config.ALPHA = deformation_hyperparameters["alpha"]
config.BETA = deformation_hyperparameters["beta"]

from dgregister.meshtransform import make_mapping

if not parserargs["folder"].endswith("/"):
    parserargs["folder"] += "/"

assert deformation_hyperparameters["starting_guess"] is None


os.makedirs(parserargs["folder"], exist_ok=True)


nx = deformation_hyperparameters["target.shape"][0]
ny = deformation_hyperparameters["target.shape"][1]
nz = deformation_hyperparameters["target.shape"][2]

cubemesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

print_overloaded("cubemesh size", nx, ny, nz)

V3 = VectorFunctionSpace(cubemesh, deformation_hyperparameters["velocity_functionspace"], deformation_hyperparameters["velocity_functiondegree"],)


# Read the control field (note: This is not yet the velocity field. Preprocessing is applied in make_mapping)
l2_control = Function(V3)

with XDMFFile(parserargs["folder"] + "Control_checkpoint.xdmf") as xdmf:
    xdmf.read_checkpoint(l2_control, "CurrentV")

mapping = make_mapping(cubemesh, control=l2_control, hyperparameters=deformation_hyperparameters)

hdf = HDF5File(cubemesh.mpi_comm(), parserargs["folder"] + parserargs["outputfilename"], "w")
hdf.write(mapping, "coordinatemapping")
hdf.close()

print_overloaded("Created vector function that maps xyz mesh coordinates")