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


# allow_ras2ras=True
# vox2ras = np.eye(4)
# vox2ras2 = np.eye(4)
# ltafile = p + "mri2fem-dataset/normalized/registered_lta/abbytoernie_ras2ras.lta"

# outvals = read_geometry(original_ventricle_file, read_metadata=True)
# metadata = outvals[2]

# ras2vox = get_surface_ras_to_image_coordinates_transform(nibabel.load(p + "mri2fem-dataset/normalized/input/abby/abby_brain.mgz"), surface_metadata=metadata)
# vox2ras = np.linalg.inv(ras2vox)
# ras2vox2 = get_surface_ras_to_image_coordinates_transform(nibabel.load(p + "mri2fem-dataset/normalized/registered/abbytoernie.mgz"), surface_metadata=metadata)
# vox2ras2 = np.linalg.inv(ras2vox2)

# for file in [p + "mri2fem-dataset/normalized/input/abby/abby_brain.mgz", p + "mri2fem-dataset/normalized/registered/abbytoernie.mgz"]:
#     print(file)
     
#     x = nibabel.load(file)
#     print(x.header)
#     print("affine")
#     print(np.round(x.affine), 0)
#     print("vox2ras-tkr")
#     print(np.round(x.header.get_vox2ras_tkr()))

#     print("Read from Yngve skript, no metadata;")
#     y = get_surface_ras_to_image_coordinates_transform(x)
#     print(np.round(y, 0))
#     print("Read from Yngve skript, with metadata;")
#     y = get_surface_ras_to_image_coordinates_transform(x, surface_metadata=metadata)
#     print(np.round(y, 0))

# breakpoint()

# translation_matrix = np.eye(4)
# translation_matrix[:3, -1] = metadata['cras']

# vox2ras = nibabel.load(p + "mri2fem-dataset/normalized/input/abby/abby_brain.mgz").header.get_vox2ras_tkr()
# vox2ras2 = nibabel.load(p + "mri2fem-dataset/normalized/input/ernie/ernie_brain.mgz").header.get_vox2ras_tkr()

# print(vox2ras)
# print(vox2ras2)

for inverse in [False]:

    newmesh = move_mesh(ltafile=ltafile, meshfile=meshfile, inverse=inverse, vox2ras=vox2ras, vox2ras2=vox2ras2, allow_ras2ras=allow_ras2ras)

    # newmesh.coordinates()[:] += metadata["cras"]

    outmeshfile = meshfile.replace(".xml", "inv" + str(inverse) + ".xml")

    File(outmeshfile) << newmesh

    mm = meshio.read(outmeshfile)

    mm.write(outmeshfile.replace(".xml", ".stl"))

    # embed()

    write_geometry(outmeshfile.replace(".xml", "_fsformat"), coords=mm.points, faces=mm.cells_dict["triangle"], create_stamp=None, volume_info=metadata)