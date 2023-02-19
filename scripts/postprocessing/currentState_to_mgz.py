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


hyperparameters = json.load(open("hyperparameters.json"))

hyperparameters["input"] = hyperparameters["input"].replace("/d1/", "/D1/")
hyperparameters["target"] = hyperparameters["target"].replace("/d1/", "/D1/")
# breakpoint()

assert os.path.isfile(hyperparameters["input"])

[nx, ny, nz] = hyperparameters["target.shape"]



# inputimage = nibabel.load(hyperparameters["input"])

path_to_inputdata = pathlib.Path(hyperparameters["input"]).parent


files = json.load(open(path_to_inputdata / "files.json"))

# breakpoint()

a = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/"
b = "/home/bastian/D1/registration/"
# 0 is abbytoernie, 1 is ernie

# breakpoint()

# inputimage = files["images"][0].replace(a, b)
# targetimage = files["images"][1].replace(a, b)

# print("targetimage=", targetimage,)

# tt=pathlib.Path(files["images"][0].replace(a, b)).parent.parent
# targetimage = nibabel.load(str(tt / "input" / "ernie" / "ernie_brain.mgz"))

targetimage = nibabel.load(hyperparameters["input"])

ernie_affine = targetimage.affine

# ernie_affine = np.eye(4)

targetimage = targetimage.get_fdata()



# assert os.path.isfile(targetimage)
#targetimage = nibabel.load(targetimage).get_fdata()





# assert np.max(originalimage.affine) != 1


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


# box_bounds = get_bounding_box_limits(box)
# filled_image = cut_to_box(image=targetimage, box_bounds=box_bounds, inverse=True, cropped_image=retimage, pad=files["npad"])# inputimage.get_fdata())

filled_image = retimage

# if not np.allclose(filled_image, originalimage.get_fdata(), rtol=1e-1, atol=1e-1):
#     print("WARING: IMAGES MAYBE NOT THE SAME?")

assert filled_image.shape == targetimage.shape

# breakpoint()

nii = nibabel.Nifti1Image(filled_image, ernie_affine)
nibabel.save(nii, "CurrentState.mgz")

nii = nibabel.Nifti1Image(np.abs(filled_image-targetimage/np.max(targetimage)), ernie_affine)
nibabel.save(nii, "AbsDiff.mgz")


# nii = nibabel.Nifti1Image(filled_image, nibabel.load(ernieimage).affine)
# nibabel.save(nii, "CurrentState_ernieAff.mgz")



# nii = nibabel.Nifti1Image(filled_image, nibabel.load("/home/bastian/D1/registration/mri2fem-dataset/processed/input/abby/abby_brain.mgz").affine)
# nibabel.save(nii, "CurrentState_AbbyAff.mgz")