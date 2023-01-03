from fenics import *
from fenics_adjoint import *
import os
import json
import time
import argparse
import numpy as np

set_log_level(LogLevel.CRITICAL)

# PETScOptions.set("mat_mumps_use_omp_threads", 8)
# PETScOptions.set("mat_mumps_icntl_35", True) # set use of BLR (Block Low-Rank) feature (0:off, 1:optimal)
# PETScOptions.set("mat_mumps_cntl_7", 1e-8) # set BLR relaxation
# PETScOptions.set("mat_mumps_icntl_4", 3)   # verbosity
# PETScOptions.set("mat_mumps_icntl_24", 1)  # detect null pivot rows
# PETScOptions.set("mat_mumps_icntl_22", 0)  # out of core
# #PETScOptions.set("mat_mumps_icntl_14", 250) # max memory increase in %

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass
        # print("passed")

print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

from mri_utils.helpers import load_velocity, get_lumped_mass_matrices, interpolate_velocity
from mri_utils.MRI2FEM import read_image

import config # import hyperparameters

parser = argparse.ArgumentParser()

parser.add_argument("--outfoldername", required=True, type=str, help=""" name of folder to store to under "path + "output_dir" """)
parser.add_argument("--code_dir", type=str, default="/home/bastian/Oscar-Image-Registration-via-Transport-Equation/")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--slurmid", type=str, required=True)
parser.add_argument("--solver", default="krylov", choices=["lu", "krylov"])
parser.add_argument("--timestepping", default="RungeKutta", choices=["RungeKutta", "CrankNicolson", "explicitEuler"])
parser.add_argument("--smoothen", default=True, action="store_true", help="Obsolete flag. Use proper scalar product")
parser.add_argument("--nosmoothen", default=False, action="store_true", help="Sets smoothen=False")

parser.add_argument("--Pic2FEN", default=False, action="store_true", help="Load images using old code")

parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--lbfgs_max_iterations", type=float, default=400)
parser.add_argument("--dt_buffer", type=float, default=0.1)
parser.add_argument("--max_timesteps", type=float, default=None)
parser.add_argument("--state_functiondegree", type=int, default=1)


parser.add_argument("--vinit", type=float, default=0)
parser.add_argument("--readname", type=str, default="-1")
parser.add_argument("--starting_guess", type=str, default=None)
# parser.add_argument("--interpolate", default=False, action="store_true", help="Interpolate coarse v to fine mesh; required if the images for --starting_guess and --input are not the same")
parser.add_argument("--debug", default=False, action="store_true", help="Debug")

parser.add_argument("--input", default="mridata_3d/091registeredto205_padded_coarsened.mgz")
parser.add_argument("--target", default="mridata_3d/205_cropped_padded_coarsened.mgz")

hyperparameters = vars(parser.parse_args())


os.chdir(hyperparameters["code_dir"])
print_overloaded("Setting pwd to", hyperparameters["code_dir"])

assert "/" not in hyperparameters["outfoldername"]

if hyperparameters["starting_guess"] is not None:
    assert os.path.isfile(hyperparameters["starting_guess"])

set_log_level(20)

if hyperparameters["nosmoothen"]:
    print_overloaded(".................................................................................................................................")
    print_overloaded("--nosmoothen is set, will not use transform and preconditioning!")
    print_overloaded(".................................................................................................................................")
    hyperparameters["smoothen"] = False

hyperparameters["preconditioner"] = "amg"
hyperparameters["outputfolder"] = hyperparameters["output_dir"] + hyperparameters["outfoldername"]
hyperparameters["lbfgs_max_iterations"] = int(hyperparameters["lbfgs_max_iterations"])
hyperparameters["MassConservation"] = False
hyperparameters["velocity_functiondegree"] = 1
hyperparameters["velocity_functionspace"] = "CG"
hyperparameters["state_functionspace"] = "DG"

config.hyperparameters = hyperparameters

print_overloaded("Setting config.hyperparameters")

for key, item in hyperparameters.items():
    print_overloaded(key, ":", item)

if not os.path.isdir(hyperparameters["outputfolder"]):
    os.makedirs(hyperparameters["outputfolder"], exist_ok=True)

if hyperparameters["starting_guess"] is not None:
    domainmesh, vCG, controlfun = load_velocity(hyperparameters, controlfun=None)
    
    if hyperparameters["interpolate"]:
        domainmesh, vCG, controlfun = interpolate_velocity(hyperparameters, domainmesh, vCG, controlfun)

    # print_overloaded("-------------------------------------------------------------------")
    # print_overloaded("Testing script, EXITING")
    # print_overloaded("-------------------------------------------------------------------")
    # exit()
else:
    # mesh will be created from first image
    domainmesh = None
    controlfun = None

if hyperparameters["Pic2FEN"]:

    from Pic2Fen import Pic2FEM
    (domainmesh, Img, NumData) = Pic2FEM(hyperparameters["input"], mesh=None)
    (mesh_goal, Img_goal, NumData_goal) = Pic2FEM(hyperparameters["target"], mesh=domainmesh)

    Space = FunctionSpace(domainmesh, "DG", 1)
    Img = project(sqrt(inner(Img, Img)), Space)
    Img.rename("img", "")
    Img_goal = project(sqrt(inner(Img_goal, Img_goal)), Space)
    NumData = 1

else:

    (domainmesh, Img, NumData) = read_image(hyperparameters, name="input", mesh=domainmesh)
    (mesh_goal, Img_goal, NumData_goal) = read_image(hyperparameters, name="target", mesh=domainmesh)


if hyperparameters["starting_guess"] is None:
    # Can now create function space after mesh is created from image
    vCG = VectorFunctionSpace(domainmesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"])

T_final = 1

h = CellDiameter(domainmesh)
h = float(assemble(h*dx))

hyperparameters["mehsh"] = h
hyperparameters["maxMeshCoordinate"] = np.max(domainmesh.coordinates())

# hyperparameters["input.shape"]
if hyperparameters["max_timesteps"] is None:
    hyperparameters["expected_distance_covered"] = 0.25 # assume that voxels need to be moved over a distance of max. 25 % of the image size.
    v_needed = hyperparameters["expected_distance_covered"] / T_final
    hyperparameters["DeltaT"] = hyperparameters["dt_buffer"] * float(h) / v_needed #1e-3
    print_overloaded("calculated initial time step size to", hyperparameters["DeltaT"])
    hyperparameters["DeltaT_init"] = hyperparameters["DeltaT"]

    hyperparameters["max_timesteps"] = int(1 / hyperparameters["DeltaT"])
else:
    hyperparameters["max_timesteps"] = int(hyperparameters["max_timesteps"])
    hyperparameters["DeltaT"] = 1 / hyperparameters["max_timesteps"]

if MPI.rank(MPI.comm_world) == 0:
    with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile, sort_keys=True, indent=4)
else:
    pass

# print_overloaded("Normalizing input and target with")
# print_overloaded("Img.vector()[:].max()", Img.vector()[:].max())
# print_overloaded("Img_goal.vector()[:].max()", Img_goal.vector()[:].max())

# Img.vector()[:] *= 1 / Img.vector()[:].max()
# Img_goal.vector()[:] *= 1 / Img_goal.vector()[:].max()


# print_overloaded("Applying ReLU() to images")
# Img.vector()[:] = np.where(Img.vector()[:] < 0, 0, Img.vector()[:])
# Img_goal.vector()[:] = np.where(Img_goal.vector()[:] < 0, 0, Img_goal.vector()[:])

# print_overloaded("Normalized:")
print_overloaded("Img.vector()[:].mean()", Img.vector()[:].mean())
print_overloaded("Img_goal.vector()[:].mean()", Img_goal.vector()[:].mean())

# inp=Img.vector()[:]
# tar=Img_goal.vector()[:]
# print_overloaded(np.sum(np.where(inp < 0, 1, 0)) / inp.size)
# print_overloaded(np.sum(np.where(tar < 0, 1, 0)) / tar.size)

# print_overloaded(np.sum(inp), np.sum(np.abs(inp)))
# print_overloaded(np.sum(tar), np.sum(np.abs(tar)))

#  breakpoint()

# output file
# fCont = XDMFFile(MPI.comm_world, hyperparameters["outputfolder"] + "/Control.xdmf")
# fCont .write(mesh, '/mesh')
# fCont.parameters["flush_output"] = True
# fCont.parameters["rewrite_function_mesh"] = False

controlFile = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Control.hdf", "w")
controlFile.write(domainmesh, "mesh")
# controlFile.parameters["rewrite_function_mesh"] = False

stateFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/State.hdf", "w")
stateFile.write(domainmesh, "mesh")
# stateFile.parameters["rewrite_function_mesh"] = False
# stateFile.parameters["flush_output"] = True
# stateFile.parameters["rewrite_function_mesh"] = False
# FOut.parameters["functions_share_mesh"] = True

velocityFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/VelocityField.hdf", "w")
velocityFile.write(domainmesh, "mesh")
# velocityFile.parameters["flush_output"] = True
# velocityFile.parameters["rewrite_function_mesh"] = False


# file = XDMFFile(MPI.comm_world, hyperparameters["outputfolder"] + "/Input.xdmf")
# file.parameters["flush_output"] = True
# file.parameters["rewrite_function_mesh"] = False
# # fCont.write(Img.function_space().mesh(), '/mesh')
# file.write(Img, 0)
# file.close()

# file = XDMFFile(MPI.comm_world, hyperparameters["outputfolder"] + "/Target.xdmf")
# file.parameters["flush_output"] = True
# file.parameters["rewrite_function_mesh"] = False
# # fCont.write(Img.function_space().mesh(), '/mesh')
# file.write(Img_goal, 0)
# file.close()


# Img_goal.vector().update_ghost_values()

with XDMFFile(hyperparameters["outputfolder"] + "/Target.xdmf") as xdmf:
    xdmf.write_checkpoint(Img_goal, "Img_goal", 0.)

with XDMFFile(hyperparameters["outputfolder"] + "/Input.xdmf") as xdmf:
    xdmf.write_checkpoint(Img, "Img", 0.)

files = {
    "velocityFile": velocityFile,
    "stateFile": stateFile,
    "controlFile":controlFile
}

Img.rename("input", "")
Img_goal.rename("target", "")
# NumData = 1

# if not hyperparameters["debug"]:
#     File(hyperparameters["outputfolder"] + "/input.pvd") << Img
#     File(hyperparameters["outputfolder"] + "/target.pvd") << Img_goal

#     print_overloaded("Wrote input and target to pvd files")

if hyperparameters["smoothen"]:
    _, M_lumped_inv = get_lumped_mass_matrices(vCG=vCG)
else:
    M_lumped_inv = None

t0 = time.time()

from mri_utils.find_velocity import find_velocity, CFLerror


files["lossfile"] = hyperparameters["outputfolder"] + '/loss.txt'
files["regularizationfile"] = hyperparameters["outputfolder"] + '/regularization.txt'

# for n in range(1):
    
#     try:
try:
    find_velocity(Img, Img_goal, vCG, M_lumped_inv, hyperparameters, files, starting_guess=controlfun)
except RuntimeError:
    print(":" * 100)
    print("Trying with LU solver")
    print(":" * 100)
    hyperparameters["solver"] = "lu"
    
    hyperparameters["krylov_failed"] = True

    with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile, sort_keys=True, indent=4)

    find_velocity(Img, Img_goal, vCG, M_lumped_inv, hyperparameters, files, starting_guess=controlfun)
    
    #     break
    # except CFLerror:

    #     raise NotImplementedError("Something went wrong here before, 'exploding' gradients-like. Need to be checked")
    #     hyperparameters["DeltaT"] *= 1 / 2
    #     print_overloaded("CFL condition violated, reducing time step size and retry")
    #     pass    
    
    # # sanity check
    # if hyperparameters["starting_guess"] is None:
    #     assert controlfun is None

tcomp = (time.time()-t0) / 3600

hyperparameters["optimization_time_hours"] = tcomp

if MPI.rank(MPI.comm_world) == 0:
    with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile, sort_keys=True, indent=4)
else:
    pass

print_overloaded("Optimize3d.py ran succesfully :-)")