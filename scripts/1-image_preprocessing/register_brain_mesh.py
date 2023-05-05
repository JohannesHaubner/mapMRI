import os, pathlib
import numpy as np
import nibabel
from dgregister.helpers import view, read_vox2vox_from_lta
from fenics import *

from nibabel.affines import apply_affine


path = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/manually_registered_brain_mesh/")

regimagepath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/registered/abbytoernie.mgz"

in_imagepath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/input/abby/abby_brain.mgz"


lta = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/registered/abbytoernie.lta"

inputpath = path / "input"

outputpath =path / "output"

os.makedirs(outputpath, exist_ok=True)

mesh = Mesh(str(inputpath / "abby16.xml"))


abby_image = nibabel.load(in_imagepath)
reg_abby_image= nibabel.load(regimagepath)

abby_affine = abby_image.affine
abby_vox2ras = abby_image.header.get_vox2ras_tkr()

regabby_affine = reg_abby_image.affine
regabby_vox2ras = reg_abby_image.header.get_vox2ras_tkr()

xyz = mesh.coordinates()


lta = read_vox2vox_from_lta(lta)

xyz_vox = apply_affine(aff=np.linalg.inv(abby_vox2ras), pts=xyz)

xyz_reg = apply_affine(aff=lta, pts=xyz_vox)

xyz_ras = apply_affine(aff=regabby_vox2ras, pts=xyz_reg)


mesh.coordinates()[:] = xyz_ras

File(str(outputpath / "abby_registered_brain_mesh.xml")) << mesh
meshio_command = "meshio-convert "
meshio_command += str(outputpath / "abby_registered_brain_mesh.xml") + " "
meshio_command += str(outputpath / "abby_registered_brain_mesh.xml").replace(".xml", ".xdmf")

os.system(meshio_command)
