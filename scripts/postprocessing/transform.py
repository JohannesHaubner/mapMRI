import json
import os
import pathlib
import nibabel
import numpy
import numpy as np
from fenics import *
# from fenics_adjoint import *
from dgregister.meshtransform import map_mesh, make_mapping

# from IPython import embed

if "home/bastian" in os.getcwd():
    resultpath = "/home/bastian/D1/registration/mriregistration_outputs/"

else:
    path_to_data = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/coarsecropped/"
    resultpath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mriregistration_outputs/"
    path_to_meshes = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/scripts/preprocessing/chp4/outs/"
    # if debug:
    #     path_to_data = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/cropped/"
    
    def path_to_wmparc(subj):

        if subj == "abby":
            # return "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/registered/abbytoernie.mgz"

            return "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/freesurfer/abby/mri/wmparc.mgz"

        elif subj == "ernie":

            return "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/freesurfer/ernie/mri/wmparc.mgz"

        else:
            raise ValueError

# # jobfoldername = "OCDRK1A1e-05LBFGS1000"
# jobfoldername = "OCDRK1A0.001LBFGS1000"
# jobfile = resultpath + jobfoldername + "/"
# velocityfilename = "opts/opt_phi_204.h5"
# readname = "function"

jobfoldername = "E100A0.0001LBFGS100NOSMOOTHEN"
jobfile = resultpath + jobfoldername + "/"
velocityfilename = "VelocityField.hdf"
readname = "-1"

hyperparameters = json.load(open(jobfile + "hyperparameters.json"))



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

print("cubemesh size", nx, ny, nz)

V3 = VectorFunctionSpace(cubemesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"],)

if os.path.isfile(mapfile):
    hdf = HDF5File(cubemesh.mpi_comm(), mapfile, "r")
    mapping = Function(V3)
    hdf.read(mapping, mappingname)
    hdf.close()

    # For debugging purpose: No transformation
    # mapping = interpolate(Expression(("x[0]", "x[1]", "x[2]"), degree=1), V3)

    assert norm(mapping) != 0
    print("Read mapping")

else:
    v = Function(V3)
    vfile = jobfile + velocityfilename
    hdf = HDF5File(cubemesh.mpi_comm(), vfile, "r")
    hdf.read(v, readname)
    hdf.close()
        
    mapping = make_mapping(cubemesh, v, jobfile, hyperparameters, ocd=ocd)

    hdf = HDF5File(cubemesh.mpi_comm(), mapfile, "w")
    hdf.write(mapping, mappingname)
    hdf.close()
    print("Created vector function that maps xyz mesh coordinates")

if "home/bastian" in os.getcwd():
    print("Done, exiting")
    exit()

xmlfile2 = path2 + subj2 + str(res) + ".xml"
xmlfile1 = path1 + subj1 + str(res) + ".xml"

imgfile1 = path_to_wmparc(subj1)
imgfile2 = path_to_wmparc(subj2)

if not (nx == 75 and ny == 79 and nz == 98):
    raise ValueError
else:
    box = np.load(path_to_data + "box.npy")
    coarsening_factor=2
    npad = 4

brainmesh2 = map_mesh(xmlfile1, imgfile1, imgfile2, mapping, box=box, 
                    outfolder=jobfile + "postprocessing/", npad=npad,
                    coarsening_factor=coarsening_factor)


inputmesh = Mesh(xmlfile1)

# embed()

# Store as xdmf file for paraview visualization
xmlfile3 = jobfile + "postprocessing/" + "transformed_input_mesh.xdmf"
with XDMFFile(xmlfile3) as xdmf:
    xdmf.write(brainmesh2) # , "mesh")

# Store as hdf File for use in further FEniCS simulation
hdf = HDF5File(brainmesh2.mpi_comm(), xmlfile3.replace(".xdmf", ".h5"), "w")
hdf.write(brainmesh2, "mesh")
hdf.close()


for m, mname in zip([inputmesh, brainmesh2],["Input mesh", "After transform"]):
    print(mname)
    print("Ratio min max", MeshQuality.radius_ratio_min_max(m))
    print("hmax",  m.hmax())
    print("hmin",  m.hmin())