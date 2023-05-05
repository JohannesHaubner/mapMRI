from fenics import *
import meshio
import nibabel
from dgregister.helpers import move_mesh, get_surface_ras_to_image_coordinates_transform
import numpy as np

from nibabel.freesurfer.io import read_geometry, write_geometry
from IPython import embed

allow_ras2ras = False


p = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/"

# original_ventricle_file = p + "mri2fem-dataset/meshes/ventricles/abby/lh.fs_ventricles" # "ventricles.stl"

meshfile = p + "mri2fem-dataset/meshes/ventricles/abby/ventricles_boundary.xml"

ltafile = p + "mri2fem-dataset/normalized/registered/abbytoernie.lta"
vox2ras = nibabel.load(p + "mri2fem-dataset/normalized/input/abby/abby_brain.mgz").affine #.header.get_vox2ras_tkr()
vox2ras2 = nibabel.load(p + "mri2fem-dataset/normalized/registered/abbytoernie.mgz").affine # header.get_vox2ras_tkr()


inverse=False
newmesh = move_mesh(ltafile=ltafile, meshfile=meshfile, inverse=inverse, vox2ras=vox2ras, vox2ras2=vox2ras2, allow_ras2ras=allow_ras2ras)

# newmesh.coordinates()[:] += metadata["cras"]

outmeshfile = meshfile.replace(".xml", "inv" + str(inverse) + ".xml")

File(outmeshfile) << newmesh

mm = meshio.read(outmeshfile)

mm.write(outmeshfile.replace(".xml", ".stl"))

# embed()

# write_geometry(outmeshfile.replace(".xml", "_fsformat"), coords=mm.points, faces=mm.cells_dict["triangle"], create_stamp=None, volume_info=metadata)