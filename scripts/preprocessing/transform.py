import json
import os
import pathlib

import nibabel
import numpy
import numpy as np
from fenics import *
from fenics_adjoint import *
from nibabel.affines import apply_affine

from dgregister.meshtransform import map_mesh, make_mapping



if "home/bastian" in os.getcwd():
    resultpath = "/home/bastian/D1/registration/mriregistration_outputs/"

else:
    path_to_data = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/coarsecropped/"
    resultpath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mriregistration_outputs/"
    def path_to_wmparc(subj):

        if subj == "abby":
            return "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/registered/abbytoernie.mgz"
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

path1 = "chp4/outs/" + subj1 + "/"
path2 = "chp4/outs/" + subj2 + "/"

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
    assert norm(mapping) != 0
    print("Read mapping")

else:
    # FName = pathlib.Path(path_to_data) / pathlib.Path(hyperparameters["input"]).name
    # FName = str(FName)

    # cubemesh, cubeimg, _ = MRI2FEM.read_image(hyperparameters={"image": FName, "state_functionspace": state_functionspace, "state_functiondegree":1}, 
    #                 name="image", mesh=None, normalize=False, threshold=False)


    # V = VectorFunctionSpace(cubemesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"],)
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

assert "abby" in imgfile1
assert "ernie" in imgfile2

box = np.load(path_to_data + "box.npy")


inputmesh = Mesh(xmlfile2)
print("targetmesh.coordinates().min = ", np.min(inputmesh.coordinates(), axis=0))
print("targetmesh.coordinates().max = ", np.max(inputmesh.coordinates(), axis=0))

targetmesh = Mesh(xmlfile2)
print("targetmesh.coordinates().min = ", np.min(targetmesh.coordinates(), axis=0))
print("targetmesh.coordinates().max = ", np.max(targetmesh.coordinates(), axis=0))
# exit()


brainmesh2 = map_mesh(xmlfile1, imgfile1, imgfile2, mapping, box=box, 
                    outfolder=jobfile + "postprocessing/",
                    coarsening_factor=2)

xmlfile3 = jobfile + "postprocessing/" + "transformed_input_mesh.xdmf"
with XDMFFile(xmlfile3) as xdmf:
    xdmf.write(brainmesh2) # , "mesh")

brainmesh = Mesh(xmlfile1)

for m, mname in zip([brainmesh, brainmesh2],["original", "transformed"]):
    print(mname, MeshQuality.radius_ratio_min_max(m))

print("paraview " + xmlfile1.replace("xml", "xdmf") + " " + xmlfile2.replace("xml", "xdmf") + " " + xmlfile3)

