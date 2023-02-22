from fenics import *
from fenics_adjoint import *
import json
from dgregister.MRI2FEM import fem2mri
import nibabel
import os, pathlib
import numpy as np
import argparse
from dgregister.helpers import cut_to_box, get_bounding_box_limits

parser = argparse.ArgumentParser()
parser.add_argument("path")
parserargs = vars(parser.parse_args())

os.chdir(parserargs["path"])


h = json.load(open("hyperparameters.json"))

assert os.path.isfile(h["input"])

[nx, ny, nz] = h["target.shape"]



inputimage = nibabel.load(h["input"])

path_to_inputdata = pathlib.Path(h["input"]).parent


files = json.load(open(path_to_inputdata / "files.json"))

a = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/"
b = "/home/bastian/D1/registration/"
# 0 is abbytoernie, 1 is ernie
# targetimage = files["images"][0].replace(a, b)
# ernieimage = files["images"][1].replace(a, b)

# breakpoint()

tt=pathlib.Path(files["images"][0].replace(a, b)).parent.parent
originalimage = nibabel.load(str(tt / "input" / "ernie" / "ernie_brain.mgz"))


# assert os.path.isfile(targetimage)
# originalimage = nibabel.load(targetimage)
# print("targetimage=", targetimage,)


if "coarsecropped" in str(path_to_inputdata):
    raise NotImplementedError

box = np.load(path_to_inputdata / "box.npy").astype(bool)




# nii = nibabel.Nifti1Image(filled_image, originalimage.affine)

# nibabel.save(nii, "/home/bastian/D1/registration/test/remapped.mgz")
# print(h["input"])
# print(targetimage, "(targetimage)")
# breakpoint()
# exit()

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

# TODO FIXME
box_bounds = get_bounding_box_limits(box)
filled_image = cut_to_box(image=originalimage.get_fdata(), box_bounds=box_bounds, inverse=True, cropped_image=retimage, pad=files["npad"])# inputimage.get_fdata())

# if not np.allclose(filled_image, originalimage.get_fdata(), rtol=1e-1, atol=1e-1):
#     print("WARING: IMAGES MAYBE NOT THE SAME?")

# breakpoint()

nii = nibabel.Nifti1Image(filled_image, originalimage.affine)
nibabel.save(nii, "CurrentState.mgz")

nii = nibabel.Nifti1Image(np.abs(filled_image-originalimage.get_fdata()/np.max(originalimage.get_fdata())), originalimage.affine)
nibabel.save(nii, "AbsDiff.mgz")


# nii = nibabel.Nifti1Image(filled_image, nibabel.load(ernieimage).affine)
# nibabel.save(nii, "CurrentState_ernieAff.mgz")



# nii = nibabel.Nifti1Image(filled_image, nibabel.load("/home/bastian/D1/registration/mri2fem-dataset/processed/input/abby/abby_brain.mgz").affine)
# nibabel.save(nii, "CurrentState_AbbyAff.mgz")