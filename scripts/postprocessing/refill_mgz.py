from fenics import *
from fenics_adjoint import *
import json
from dgregister.MRI2FEM import fem2mri
import nibabel
import os, pathlib
import numpy as np
import argparse
from dgregister.helpers import cut_to_box

parser = argparse.ArgumentParser()
parser.add_argument("path")
parserargs = vars(parser.parse_args())

os.chdir(parserargs["path"])


h = json.load(open("hyperparameters.json"))

a = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/"
b = "/home/bastian/D1/registration/"

h["input"] = h["input"].replace(b, a)

assert os.path.isfile(h["input"])

[nx, ny, nz] = h["input.shape"]

inputimage = nibabel.load(h["input"])

path_to_inputdata = pathlib.Path(h["input"]).parent


files = json.load(open(path_to_inputdata / "files.json"))


# 0 is abbytoernie, 1 is ernie
inputimage = files["images"][0].replace(b, a)
ernieimage = files["images"][1].replace(b, a)

assert os.path.isfile(inputimage)
originalimage = nibabel.load(inputimage)
print("targetimage=", inputimage,)


if "coarsecropped" in str(path_to_inputdata):
    raise NotImplementedError

box = np.load(path_to_inputdata / "box.npy").astype(bool)
# exit()



retimage = nibabel.load("Finalstate.mgz").get_fdata()

filled_image = cut_to_box(image=originalimage.get_fdata(), box=box, inverse=True, cropped_image=retimage)# inputimage.get_fdata())

# if not np.allclose(filled_image, originalimage.get_fdata(), rtol=1e-1, atol=1e-1):
#     print("WARING: IMAGES MAYBE NOT THE SAME?")

# breakpoint()

nii = nibabel.Nifti1Image(filled_image, originalimage.affine)
nibabel.save(nii, "Finalstate_filled.mgz")

nii = nibabel.Nifti1Image(np.abs(filled_image-originalimage.get_fdata()/np.max(originalimage.get_fdata())), originalimage.affine)
nibabel.save(nii, "AbsDiff.mgz")


# nii = nibabel.Nifti1Image(filled_image, nibabel.load(ernieimage).affine)
# nibabel.save(nii, "CurrentState_ernieAff.mgz")



# nii = nibabel.Nifti1Image(filled_image, nibabel.load("/home/bastian/D1/registration/mri2fem-dataset/processed/input/abby/abby_brain.mgz").affine)
# nibabel.save(nii, "CurrentState_AbbyAff.mgz")