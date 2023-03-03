from fenics import *
import meshio
import numpy as np
import os, subprocess
from freesurfer_surface import Surface, Vertex, Triangle
import pathlib

superpath = pathlib.Path("programming/Oscar-Image-Registration-via-Transport-Equation/registration/")

inputpath = superpath / "transported-meshes"

outpath = inputpath

meshfiles = [inputpath / x for x in os.listdir(inputpath) if x.endswith(".xml")]

if len(meshfiles) > 1:
    raise ValueError

else:
    xmlfile = meshfiles[0]

# outpath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/"
# outpath += "croppedmriregistration_outputs/E100A0.01LBFGS100/postprocessing_newer/"
# inputpath = outpath
# meshname = "transformed_input_mesh"
surfacefile = "/home/basti/Downloads/lh.white"
# xmlfile = inputpath + meshname + ".xml"
subj = "ernie"

# p = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/freesurfer/" + subj + "/surf/"
# for f in os.listdir(p):
#     try:
#         surface = Surface.read_triangular(p + f)
#         print("Read,", f)
#         info = surface.volume_geometry_info
#         print(surface.volume_geometry_info)

#         break
#     except:
#         pass

outfile = xmlfile.replace(".xml", "")

if os.path.isfile(outfile):
    print("*"*80)
    print("File already exists, not recomputing")
    print("*"*80)
    command = "freeview"
    command += " /home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/input/abby/abby_brain.mgz"
    command += " /home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/input/ernie/ernie_brain.mgz"
    command += " -f " + outfile
    command += " /home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/scripts/preprocessing/chp4/outs/abby/abby16:edgecolor=magenta"
    command += " /home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/scripts/preprocessing/chp4/outs/ernie/ernie16:edgecolor=blue"
    print()
    subprocess.run(command, shell=True)
    exit()



assert os.path.isfile(xmlfile)

try:
    brainmesh = Mesh(xmlfile)
except:
    brainmesh = Mesh()
    hdf = HDF5File(brainmesh.mpi_comm(), xmlfile.replace(".xml", ".h5"), "r")
    hdf.read(brainmesh, "/mesh", False)
    hdf.close()

if "boundary" in xmlfile:
    bmesh = brainmesh
else:
    bmesh = BoundaryMesh(brainmesh, 'exterior')


xyz = bmesh.coordinates()

surface = Surface.read_triangular(surfacefile)

surface.vertices = []
surface.triangles = []

surface.volume_geometry_info = info

# embed()
# exit()

for idc in range(bmesh.cells().shape[0]):
    i1, i2, i3 = bmesh.cells()[idc, :]

    x1 = xyz[i1, :]
    x2 = xyz[i2, :]
    x3 = xyz[i3, :]

    vertex_a = surface.add_vertex(Vertex(x1[0], x1[1], x1[2]))
    vertex_b = surface.add_vertex(Vertex(x2[0], x2[1], x2[2]))
    vertex_c = surface.add_vertex(Vertex(x3[0], x3[1], x3[2]))
    surface.triangles.append(Triangle((vertex_a, vertex_b, vertex_c)))

surface.write_triangular(outfile)
