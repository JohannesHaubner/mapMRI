import argparse
import json
import os
import numpy as np
from fenics import *
# from fenics_adjoint import *
import h5py
import meshio

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass



parser = argparse.ArgumentParser()
parser.add_argument("--mapping_only", help="only create the coordinate mapping and exit (no mesh transform)", action="store_true", default=False)
parser.add_argument("--folder", type=str, default="mriregistration_outputs")
parser.add_argument("--job", type=str, required=True)

parser.add_argument("--velocityfilename", type=str, default="VelocityField.hdf") #opts/opt_phi_6.h5")
parser.add_argument("--readname", type=str, default="-1")#"function")

parserargs = vars(parser.parse_args())
# from IPython import embed




if parserargs["mapping_only"]:
    print_overloaded("--mapping_only is set, will only create the mapping and exit()")

if "home/bastian" in os.getcwd():
    path_to_stuff = "/home/bastian/D1/registration/"
    resultpath = path_to_stuff
    path_to_data = path_to_stuff
    path_to_meshes = "/home/bastian/"
else:
    path_to_stuff = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/"
    path_to_data = path_to_stuff
    resultpath = path_to_stuff
    path_to_meshes = "/home/basti/programming/"

def path_to_wmparc(subj):

    if subj == "abby":
        return path_to_stuff + "mri2fem-dataset/freesurfer/abby/mri/wmparc.mgz"
    elif subj == "ernie":
        return path_to_stuff + "mri2fem-dataset/freesurfer/ernie/mri/wmparc.mgz"

    else:
        raise ValueError

if not parserargs["folder"].endswith("/"):
    parserargs["folder"] += "/"

resultpath += parserargs["folder"]
path_to_data += "mri2fem-dataset/processed/coarsecropped/"
path_to_meshes += "Oscar-Image-Registration-via-Transport-Equation/scripts/preprocessing/chp4/outs/"

jobfoldername = parserargs["job"]

jobfile = resultpath + jobfoldername + "/"

velocityfilename = parserargs["velocityfilename"]
readname = parserargs["readname"]

velocityfile = jobfile + velocityfilename

assert os.path.isfile(velocityfile)

if not parserargs["readname"] in list(h5py.File(velocityfile).keys()):
    raise ValueError(parserargs["readname"] + "not in keys of velocityfile:" + str(list(h5py.File(velocityfile).keys())))

# jobfoldername = "E100A0.0001LBFGS100NOSMOOTHEN"
# jobfile = resultpath + jobfoldername + "/"
# velocityfilename = "VelocityField.hdf"
# readname = "-1"

hyperparameters = json.load(open(jobfile + "hyperparameters.json"))

import dgregister.config as config
config.hyperparameters = {"optimize": False}

if hyperparameters["smoothen"]:
    config.hyperparameters = {"optimize": True}
    from fenics_adjoint import *

from dgregister.meshtransform import map_mesh, make_mapping


hyperparameters = {**parserargs, **hyperparameters}

config.hyperparameters = {**hyperparameters, **config.hyperparameters}


if "OCD" not in jobfile:
    ocd = False
    assert hyperparameters["max_timesteps"] > 1
else:
    ocd = True
    assert hyperparameters["max_timesteps"] == 1



res = 16

subj1 = "abby"
subj2 = "ernie"
# subj2 = "abby"

path1 = path_to_meshes + subj1 + "/"
path2 = path_to_meshes + subj2 + "/"

os.makedirs(jobfile + "postprocessing/", exist_ok=True)

mapfile = jobfile + "postprocessing/" + "all" + ".hdf"
mappingname = "coordinatemapping"

nx = hyperparameters["input.shape"][0]
ny = hyperparameters["input.shape"][1]
nz = hyperparameters["input.shape"][2]

cubemesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

print_overloaded("cubemesh size", nx, ny, nz)

V3 = VectorFunctionSpace(cubemesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"],)

if os.path.isfile(mapfile):
    hdf = HDF5File(cubemesh.mpi_comm(), mapfile, "r")
    mapping = Function(V3)
    hdf.read(mapping, mappingname)
    hdf.close()

    # For debugging purpose: No transformation
    # mapping = interpolate(Expression(("x[0]", "x[1]", "x[2]"), degree=1), V3)

    assert norm(mapping) != 0
    print_overloaded("Read mapping")

else:
    v = Function(V3)
    hdf = HDF5File(cubemesh.mpi_comm(), velocityfile, "r")
    hdf.read(v, readname)
    hdf.close()
        
    mapping = make_mapping(cubemesh, v, jobfile, hyperparameters, ocd=ocd)

    hdf = HDF5File(cubemesh.mpi_comm(), mapfile, "w")
    hdf.write(mapping, mappingname)
    hdf.close()
    print_overloaded("Created vector function that maps xyz mesh coordinates")

if parserargs["mapping_only"]:
    print_overloaded("--mapping_only is set, created the mapping. exit()")
    exit()

xmlfile2 = path2 + subj2 + str(res) + ".xml"
xmlfile1 = path1 + subj1 + str(res) + ".xml"

imgfile1 = path_to_wmparc(subj1)
imgfile2 = path_to_wmparc(subj2)

box = np.load(path_to_data + "box.npy")

if not (nx == 75 and ny == 79 and nz == 98):
    assert "croppedmriregistration_outputs/" == parserargs["folder"]
    coarsening_factor = 1
    npad = 0
else:
    assert "mriregistration_outputs/" == parserargs["folder"]
    coarsening_factor = 2
    npad = 4

raise_errors = True

if ocd:
    raise_errors = False

# raise_errors = False

brainmesh2 = map_mesh(xmlfile1, imgfile1, imgfile2, mapping, box=box, 
                    outfolder=jobfile + "postprocessing/", npad=npad, raise_errors=raise_errors,
                    coarsening_factor=coarsening_factor)


inputmesh = Mesh(xmlfile1)

# embed()

# # Store as xdmf file for paraview visualization
xmlfile3 = jobfile + "postprocessing/" + "transformed_input_mesh.xml"
# with XDMFFile(xmlfile3) as xdmf:
#     xdmf.write(brainmesh2) # , "mesh")

File(xmlfile3) << brainmesh2

# os.system("conda activate mri_inverse ; meshio-convert " + xmlfile3 + " " + xmlfile3.replace(".xml", ".xdmf"))

transormed_xmlmesh = meshio.read(xmlfile3)
transormed_xmlmesh.write(xmlfile3.replace(".xml", ".xdmf"))

# Store as hdf File for use in further FEniCS simulation
hdf = HDF5File(brainmesh2.mpi_comm(), xmlfile3.replace(".xdmf", ".h5"), "w")
hdf.write(brainmesh2, "mesh")
hdf.close()


for m, mname in zip([inputmesh, brainmesh2],["Input mesh", "After transform"]):
    print_overloaded(mname)
    print_overloaded("Ratio min max", MeshQuality.radius_ratio_min_max(m))
    print_overloaded("hmax",  m.hmax())
    print_overloaded("hmin",  m.hmin())