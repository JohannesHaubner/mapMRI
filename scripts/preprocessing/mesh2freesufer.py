from fenics import *
import meshio
import numpy as np
import os, subprocess
# from IPython import embed


outpath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/"
outpath += "croppedmriregistration_outputs/E100A0.01LBFGS100/postprocessing_newer/"
inputpath = outpath
meshname = "transformed_input_mesh"

# outpath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/scripts/preprocessing/chp4/outs/ernie/"
# inputpath = outpath
# meshname = "ernie16"

# subj = "ernie"
# path = "chp4/outs/" + subj + "/"
# res = 16
xmlfile = inputpath + meshname + ".xml"
# path + subj + str(res) + ".xml"
# meshfile = inputpath  # path + subj + str(res) + ".mesh"

subj = "ernie"



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


# sf = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/freesurfer/ernie/surf/lh.orig"

assert os.path.isfile(xmlfile)

try:
    brainmesh = Mesh(xmlfile)
except:
    brainmesh = Mesh()
    hdf = HDF5File(brainmesh.mpi_comm(), xmlfile.replace(".xml", ".h5"), "r")
    hdf.read(brainmesh, "/mesh", False)
    hdf.close()


# mesh = meshio.read(meshfile)

# # Extract subdomains and boundaries between regions
# # into appropriate containers
# points = mesh.points
# cells = {"triangle": mesh.cells_dict["triangle"]}
# lines =     {"line": mesh.cells_dict["line"]}
# boundaries = {"edges": [mesh.cell_data_dict["medit:ref"]["line"]]}
# subdomains = {"triangles": [mesh.cell_data_dict["medit:ref"]["triangle"]]}


# # Write the boundaries/interfaces of the mesh
# xdmf = meshio.Mesh(points, lines ,cell_data=boundaries)
# meshio.write(path + "boundaries.xdmf", xdmf)


# n = brainmesh.topology().dim()
# boundaries = MeshFunction("size_t", brainmesh, n-1, 0)
# with XDMFFile(path + "boundaries.xdmf") as infile:
#     infile.read(boundaries)#, "boundaries")

bmesh = BoundaryMesh(brainmesh, 'exterior')

# bmesh2 = Mesh(path + "boundaries.xml")

# cells = bmesh.cells()




# embed()
# exit()

xyz = bmesh.coordinates()

from freesurfer_surface import Surface, Vertex, Triangle


# sf = path + 'lh.pial'
# sf = path + 'lh.white'



p = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/freesurfer/" + subj + "/surf/"
# p = "/home/basti/Dropbox (UiO)/Sleep/228/surf/"
for f in os.listdir(p):
    try:
        surface = Surface.read_triangular(p + f)
        print("Read,", f)
        info = surface.volume_geometry_info
        print(surface.volume_geometry_info)

        break
    except:
        pass


# exit()
# surface = Surface.read_triangular(sf)

surface = Surface.read_triangular("/home/basti/Downloads/lh.white")

print(type(surface.vertices), type(surface.triangles))

surface.vertices = []
surface.triangles = [] # None

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

    # vertex_a = surface.add_vertex(Vertex(0.0, 0.0, 0.0))
    # vertex_b = surface.add_vertex(Vertex(1.0, 1.0, 1.0))
    # vertex_c = surface.add_vertex(Vertex(2.0, 2.0, 2.0))





# outfile = xmlfile.replace(".xml", "")

surface.write_triangular(outfile)
