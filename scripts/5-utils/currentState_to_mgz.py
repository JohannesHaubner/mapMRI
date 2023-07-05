"""
Convert DG0 representation of image back to voxel format and store in MRI format with affine for visualization in freeview
"""


from fenics import *
from fenics_adjoint import *
import json
from dgregister.MRI2FEM import fem2mri
import nibabel
import os, pathlib
import numpy as np
import argparse
from dgregister.helpers import cut_to_box, get_bounding_box_limits
from dgregister.helpers import crop_to_original, Meshdata

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--statename", type=str, default="CurrentState", choices=["CurrentState", "Finalstate", "Img"])
parser.add_argument("--readname", type=str, default="CurrentState", choices=["CurrentState", "Finalstate", "Img"])
parser.add_argument("--xdmffile", type=str, default="State_checkpoint.xdmf", choices=["CurrentState.xdmf" ,"State_checkpoint.xdmf", "Finalstate.xdmf", "Input.xdmf"])


parserargs = vars(parser.parse_args())

parserargs["recompute"] = True

statename = parserargs["statename"]
# os.chdir(parserargs["path"])

h = json.load(open(pathlib.Path(parserargs["path"]) / "hyperparameters.json"))

[nx, ny, nz] = h["target.shape"]


print("Initializing mesh")
mesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

print("Initializing FunctionSpace")
V = FunctionSpace(mesh, "DG", 1)

u = Function(V)

with XDMFFile(str(pathlib.Path(parserargs["path"]) / parserargs["xdmffile"])) as xdmf:
    xdmf.read_checkpoint(u, parserargs["readname"])

print("Read ", parserargs["readname"],  ", calling fem2mri")
retimage = fem2mri(function=u, shape=[nx, ny, nz])

np.save(statename+".npy", retimage)
print("Stored " + statename + ".npy")


data =  Meshdata()

aff3= data.affine
box = data.box
space = data.space
pad = data.pad

cropped_image = retimage

print(np.round(aff3, decimals=0))

filled_image = crop_to_original(orig_image=np.zeros((256, 256, 256)), cropped_image=cropped_image, box=box, space=space, pad=pad)

nii = nibabel.Nifti1Image(filled_image, aff3)
nibabel.save(nii,str(pathlib.Path(parserargs["path"]) / (statename+".mgz")))

