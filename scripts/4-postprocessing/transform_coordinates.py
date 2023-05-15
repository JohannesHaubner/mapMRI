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
parser.add_argument("--outputfoldername", type=str, default="meshtransform/")
parserargs = vars(parser.parse_args())


deformation_hyperparameter = json.load(open(parserargs["folder"] + "hyperparameters.json"))

import dgregister.config as config

config.ALPHA = deformation_hyperparameter["alpha"]
config.BETA = deformation_hyperparameter["beta"]

from dgregister.meshtransform import make_mapping

if not parserargs["folder"].endswith("/"):
    parserargs["folder"] += "/"

if not parserargs["outputfoldername"].endswith("/"):
    parserargs["outputfoldername"] += "/"


assert parserargs["outputfoldername"][0] != "/"

assert deformation_hyperparameter["starting_guess"] is None

outputfolder = str(pathlib.Path(parserargs["folder"], parserargs["outputfoldername"]))

if not outputfolder.endswith("/"):
    outputfolder += "/"

os.makedirs(outputfolder, exist_ok=True)


nx = deformation_hyperparameter["target.shape"][0]
ny = deformation_hyperparameter["target.shape"][1]
nz = deformation_hyperparameter["target.shape"][2]

cubemesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

print_overloaded("cubemesh size", nx, ny, nz)

V3 = VectorFunctionSpace(cubemesh, deformation_hyperparameter["velocity_functionspace"], deformation_hyperparameter["velocity_functiondegree"],)


# Read the control field (note: This is not yet the velocity field. Preprocessing is applied in make_mapping)
l2_control = Function(V3)

with XDMFFile(parserargs["folder"] + "/Control_checkpoint.xdmf") as xdmf:
    xdmf.read_checkpoint(l2_control, "CurrentV")

mapping = make_mapping(cubemesh, control=l2_control, hyperparameters=deformation_hyperparameter)

hdf = HDF5File(cubemesh.mpi_comm(), outputfolder + "all" + ".hdf", "w")
hdf.write(mapping, "coordinatemapping")
hdf.close()

print_overloaded("Created vector function that maps xyz mesh coordinates")