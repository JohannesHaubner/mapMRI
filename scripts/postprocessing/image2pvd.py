from fenics import *
from nibabel.affines import apply_affine
import os
import nibabel
import argparse
import json
import numpy as np
from dgregister.MRI2FEM import read_image

parser = argparse.ArgumentParser()
parser.add_argument("imagefile")
parserargs = vars(parser.parse_args())

imagefile = parserargs["imagefile"] # "/home/bastian/D1/registration/" + "mri2fem-dataset/normalized/input/ernie/" + "ernie_brain.mgz"

image = nibabel.load(imagefile)
nx, ny, nz = image.get_fdata().shape[0], image.get_fdata().shape[1], image.get_fdata().shape[2]

mesh1 = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

hyperparameters = {"smoothen_image": False}


_, u_data, _, _ = read_image(imagefile, name=None, mesh=mesh1, printout=True, threshold=False, projector=None,
                                state_functionspace="DG", state_functiondegree=0, hyperparameters=hyperparameters)

vox2ras = image.header.get_vox2ras_tkr()

print("Creating second mesh")
mesh2 = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

mesh2.coordinates()[:] = apply_affine(vox2ras, mesh1.coordinates())

u = Function(FunctionSpace(mesh2, "DG", 0))
print("Created function on transformed mesh")

u.vector()[:] = u_data.vector()[:]

File(imagefile.replace(".mgz", ".pvd")) << u

print("Stored, done")