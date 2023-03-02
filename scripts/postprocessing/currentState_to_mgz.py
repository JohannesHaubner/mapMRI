from fenics import *
from fenics_adjoint import *
import json
from dgregister.MRI2FEM import fem2mri
import nibabel
import os, pathlib
import numpy as np
import argparse
from dgregister.helpers import cut_to_box, get_bounding_box_limits
from dgregister.helpers import crop_to_original, Data

parser = argparse.ArgumentParser()
parser.add_argument("path")
# parser.add_argument("--limits", type=int, default=0, choices=[0, 2])
# parser.add_argument("--npad", type=int, choices=[0, 2])
# parser.add_argument("--box", type=str)
# parser.add_argument("--originaltarget", type=str)

parserargs = vars(parser.parse_args())

os.chdir(parserargs["path"])

h = json.load(open("hyperparameters.json"))

# assert os.path.isfile(h["input"])

[nx, ny, nz] = h["target.shape"]



if not os.path.isfile("CurrentState.npy"):
    mesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

    V = FunctionSpace(mesh, "DG", 1)

    u = Function(V)

    with XDMFFile("State_checkpoint.xdmf") as xdmf:
        xdmf.read_checkpoint(u, "CurrentState")
    retimage = fem2mri(function=u, shape=[nx, ny, nz])

    np.save("CurrentState.npy", retimage)

else:
    retimage =  np.load("CurrentState.npy")

data =  Data(input=h["input"], target=h["target"])

aff3= data.affine
box = data.box
space = data.space
pad = data.pad

cropped_image = retimage

print(np.round(aff3, decimals=0))
# exit()

filled_image = crop_to_original(orig_image=np.zeros((256, 256, 256)), cropped_image=cropped_image, box=box, space=space, pad=pad)

# nii = nibabel.Nifti1Image(filled_image, aff1)
# nibabel.save(nii, "CurrentState.mgz")

# nii = nibabel.Nifti1Image(filled_image, aff2)
# nibabel.save(nii, "CurrentState2.mgz")

nii = nibabel.Nifti1Image(filled_image, aff3)
nibabel.save(nii, "CurrentState.mgz")