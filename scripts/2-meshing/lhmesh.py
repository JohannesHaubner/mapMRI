from fenics import *
import SVMTK as svmtk
import nibabel
from nibabel.affines import apply_affine
import pathlib
import numpy as np
import meshio
from dgregister.helpers import read_vox2vox_from_lta


## NOTE lh.pial.stl can be created from lh.pial using mris_convert
# NOTE
stlfile = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/lh_mesh_verycoarse/lh.pial.stl"

fixedstlfile = stlfile.replace(".stl", "_fixed.stl")

surface = svmtk.Surface(stlfile)

# Remesh surface
surface.isotropic_remeshing(3, 1, False)

surface.smooth_taubin(5)

surface.fill_holes()

# Separate narrow gaps
# Default argument is -0.33. 
surface.separate_narrow_gaps(-0.33)
surface.smooth_taubin(1)
surface.separate_narrow_gaps(-0.33)
surface.smooth_taubin(1)

surface.save(fixedstlfile)


mm = meshio.read(fixedstlfile)
mm.write(fixedstlfile.replace(".stl", ".xml"))
mm.write(fixedstlfile.replace(".stl", ".xdmf"))


meshfile = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/lh_mesh_verycoarse/lh.pial_fixed.xml"

outputpath = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/lh_registered_verycoarse/")

regimagepath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/registered/abbytoernie.mgz"

in_imagepath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/input/abby/abby_brain.mgz"

lta = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/registered/abbytoernie.lta"


mesh = Mesh(meshfile)

print("mesh.shape", mesh.coordinates().shape)


abby_image = nibabel.load(in_imagepath)
reg_abby_image= nibabel.load(regimagepath)

abby_affine = abby_image.affine
abby_vox2ras = abby_image.header.get_vox2ras_tkr()

regabby_affine = reg_abby_image.affine
regabby_vox2ras = reg_abby_image.header.get_vox2ras_tkr()

xyz = mesh.coordinates()


lta = read_vox2vox_from_lta(lta)

## Can also try with 
# /home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/registered_lta/abbytoernie_ras2ras.lta


xyz_vox = apply_affine(aff=np.linalg.inv(abby_vox2ras), pts=xyz)

# xyz_reg = xyz_vox
xyz_reg = apply_affine(aff=lta, pts=xyz_vox)

xyz_ras = apply_affine(aff=regabby_vox2ras, pts=xyz_reg)


mesh.coordinates()[:] = xyz_ras

File(str(outputpath / "lh.xml")) << mesh

mm = meshio.read(outputpath / "lh.xml")

mm.write(outputpath / "lh.stl")
mm.write(outputpath / "lh.xdmf")