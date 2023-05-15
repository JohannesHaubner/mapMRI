from fenics import *
import os, pathlib
import numpy as np
import nibabel
from dgregister.helpers import read_vox2vox_from_lta
from nibabel.affines import apply_affine


path = pathlib.Path("./data/meshes/abby/manually_registered_brain_mesh/")

regimagepath = "./data/normalized/registered/abbytoernie.mgz"

in_imagepath = "./data/normalized/input/abby/abby_brain.mgz"

lta = "./data/normalized/registered/abbytoernie.lta"

inputmesh = "./data/meshes/abby/brain/abby8.xml"

assert os.path.isfile(inputmesh)

mesh = Mesh(inputmesh)

outputpath = path / "output"

os.makedirs(outputpath, exist_ok=True)

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
