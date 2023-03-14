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
# parser.add_argument("--recompute", action="store_true", default=False)
parser.add_argument("--statename", type=str, default="CurrentState", choices=["CurrentState", "Finalstate"])
parser.add_argument("--readname", type=str, default="CurrentState", choices=["CurrentState", "Finalstate"])
parser.add_argument("--xdmffile", type=str, default="State_checkpoint.xdmf", choices=["State_checkpoint.xdmf", "Finalstate.xdmf"])


parserargs = vars(parser.parse_args())

parserargs["recompute"] = True

statename = parserargs["statename"]
os.chdir(parserargs["path"])

h = json.load(open("hyperparameters.json"))

# assert os.path.isfile(h["input"])

[nx, ny, nz] = h["target.shape"]



if (not os.path.isfile(statename + ".npy")) or parserargs["recompute"]:
    print("Initializing mesh")
    mesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

    print("Initializing FunctionSpace")
    V = FunctionSpace(mesh, "DG", 1)

    u = Function(V)

    with XDMFFile(parserargs["xdmffile"]) as xdmf:
        xdmf.read_checkpoint(u, parserargs["readname"])

    print("Read State_checkpoint, calling fem2mri")
    retimage = fem2mri(function=u, shape=[nx, ny, nz])

    np.save(statename+".npy", retimage)
    print("Stored " + statename + ".npy")

else:
    retimage =  np.load(statename+ ".npy")

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
nibabel.save(nii, statename+".mgz")

target_aff = nibabel.load(data.original_target).affine
nii = nibabel.Nifti1Image(filled_image, target_aff)
nibabel.save(nii, statename+"_targetaff.mgz")


input_aff = nibabel.load(data.original_target.replace("ernie", "abby")).affine
nii = nibabel.Nifti1Image(filled_image, input_aff)
nibabel.save(nii, statename+"_inputaff.mgz")