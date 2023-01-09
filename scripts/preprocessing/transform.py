import json
import os
import pathlib

import nibabel
import numpy
import numpy as np
from fenics import *
from fenics_adjoint import *
from nibabel.affines import apply_affine
from dgregister import MRI2FEM, DGTransport, find_velocity_ocd, mesh_tools
from mask_mri import get_bounding_box

# from IPython import embed


from dgregister.meshtransform import map_meshes

def path_to_wmparc(subj):
    return "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/freesurfer/" + subj + "/mri/wmparc.mgz"


path_to_data = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/coarsecropped/"
resultpath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mriregistration_outputs/"

# jobfoldername = "OCDRK1A1e-05LBFGS1000"
jobfoldername = "OCDRK1A0.001LBFGS1000"
jobfile = resultpath + jobfoldername + "/"
velocityfilename = "opts/opt_phi_204.h5"
res = 16

if __name__ == "__main__":
    pass

hyperparameters = json.load(open(jobfile + "hyperparameters.json"))

FName = pathlib.Path(path_to_data) / pathlib.Path(hyperparameters["input"]).name
FName = str(FName)

if "OCD" not in jobfile:
    raise NotImplementedError

res = 16

subj1 = "abby"
subj2 = "ernie"

path1 = "chp4/outs/" + subj1 + "/"




state_functionspace = "DG"


mapfile = jobfile + "postprocessing/" + "all" + ".hdf"
mappingname = "coordinatemapping"

nx = hyperparameters["input.shape"][0]
ny = hyperparameters["input.shape"][1]
nz = hyperparameters["input.shape"][2]

cubemesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

V3 = VectorFunctionSpace(cubemesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"],)

if os.path.isfile(mapfile):
    hdf = HDF5File(cubemesh.mpi_comm(), mapfile, "r")
    mapping = Function(V3)
    hdf.read(mapping, mappingname)
    hdf.close()
    assert norm(mapping) != 0
    print("Read mapping")

else:

    # cubemesh, cubeimg, _ = MRI2FEM.read_image(hyperparameters={"image": FName, "state_functionspace": state_functionspace, "state_functiondegree":1}, 
    #                 name="image", mesh=None, normalize=False, threshold=False)


    # V = VectorFunctionSpace(cubemesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"],)


    V1 = FunctionSpace(cubemesh, "CG", 1)

    os.makedirs(jobfile + "postprocessing/", exist_ok=True)
    unity2 = Function(V1) #¤ cubeimg.function_space())
    unity2.vector()[:] = 0

    mappings = []

    for coordinate in ["x[0]", "x[1]", "x[2]"]:

        if os.path.isfile(jobfile + "postprocessing/" + coordinate + ".hdf"):

            coordinate_mapping = Function(V1)

            hdf = HDF5File(cubemesh.mpi_comm(), jobfile + "postprocessing/" + coordinate + ".hdf", "r")
            hdf.read(coordinate_mapping, "out")
            hdf.close()

            assert norm(coordinate_mapping) != 0

            mappings.append(coordinate_mapping)

        else:

            xin = interpolate(Expression(coordinate, degree=1), V1) # cubeimg.function_space())

            v = Function(V3)
            vfile = jobfile + velocityfilename
            hdf = HDF5File(cubemesh.mpi_comm(), vfile, "r")
            hdf.read(v, "function")

            hdf.close()

            print("Inverting velocity for backtransport")
            v.vector()[:] *= (-1)

            assert norm(v) > 0

            print("Transporting, ", coordinate, "coordinate")

            xout, phi_c0 = find_velocity_ocd.find_velocity(Img=xin, Img_goal=unity2, hyperparameters=hyperparameters, phi_eval=v, projection=False)

            assert norm(xout) != 0

            mappings.append(xout)

            assert os.path.isdir(jobfile + "postprocessing/")
            # xout = xin

            hdf = HDF5File(cubemesh.mpi_comm(), jobfile + "postprocessing/" + coordinate + ".hdf", "w")
            # hdf.write(cubemesh, "mesh")
            hdf.write(xin, "in")
            hdf.write(xout, "out")
            hdf.close()

            with XDMFFile(jobfile + "postprocessing/" + coordinate + "_in.xdmf") as xdmf:
                xdmf.write_checkpoint(xin, "xin", 0.)


            with XDMFFile(jobfile + "postprocessing/" + coordinate + "_out.xdmf") as xdmf:
                xdmf.write_checkpoint(xout, "xout", 0.)

    assert len(mappings) == 3

    assert True not in [norm(x) == 0 for x in mappings]

    # for coordinate in ["x[0]", "x[1]", "x[2]"]:
    vxyz = Function(V3)

    arr = np.zeros_like(vxyz.vector()[:])

    arr[0::3] = mappings[0].vector()[:]
    arr[1::3] = mappings[1].vector()[:]
    arr[2::3] = mappings[2].vector()[:]

    assert np.sum(arr) != 0

    vxyz.vector()[:] = arr
    
    mapping = vxyz

    assert norm(vxyz) != 0

    hdf = HDF5File(cubemesh.mpi_comm(), mapfile, "w")
    hdf.write(vxyz, mappingname)
    hdf.close()
    print("Created vector function that maps xyz mesh coordinates")


xmlfile2 = path1 + subj2 + str(res) + ".xml"
xmlfile1 = path1 + subj1 + str(res) + ".xml"

imgfile1 = path_to_wmparc(subj1)
imgfile2 = path_to_wmparc(subj2) 
box = np.load(path_to_data + "box.npy")

xmlfile3 = jobfile + "postprocessing/" + "transformed_input_mesh.xdmf"
with XDMFFile(xmlfile3) as xdmf:
    xdmf.write(brainmesh2) # , "mesh")

for m, mname in zip([brainmesh, brainmesh2],["original", "transformed"]):
    print(mname, MeshQuality.radius_ratio_min_max(m))

print("paraview " + xmlfile1.replace("xml", "xdmf") + " " + xmlfile2.replace("xml", "xdmf") + " " + xmlfile3)

