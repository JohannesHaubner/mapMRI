from dolfin import *
from dolfin_adjoint import *
import os
import json
import time
import argparse
import numpy as np

from mri_utils.helpers import load_velocity, get_lumped_mass_matrix
from mri_utils.MRI2FEM import read_image
from mri_utils.find_velocity import find_velocity, CFLerror



parser = argparse.ArgumentParser()

parser.add_argument("--outfolder", required=True, type=str, help=""" name of folder to store to under "path + "outputs/" """)
parser.add_argument("--code_dir", type=str, default="/home/bastian/Oscar-Image-Registration-via-Transport-Equation/")
parser.add_argument("--solver", default="lu", choices=["lu", "krylov"])
parser.add_argument("--timestepping", default="Crank-Nicolson", choices=["CrankNicolson", "explicitEuler"])
parser.add_argument("--smoothen", default=False, action="store_true", help="Use proper scalar product")
parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--lbfgs_max_iterations", type=float, default=400)
parser.add_argument("--readname", type=str)
parser.add_argument("--starting_guess", type=str, default=None)
parser.add_argument("--interpolate", default=False, action="store_true", help="Interpolate coarse v to fine mesh; required if the images for --starting_guess and --input are not the same")

parser.add_argument("--input", default="mridata_3d/091registeredto205_padded_coarsened.mgz")
parser.add_argument("--target", default="mridata_3d/205_cropped_padded_coarsened.mgz")

hyperparameters = vars(parser.parse_args())

for key, item in hyperparameters.items():
    print(key, ":", item)

os.chdir(hyperparameters["code_dir"])
print("Setting pwd to", hyperparameters["code_dir"])

assert "/" not in hyperparameters["outfolder"]

set_log_level(20)

hyperparameters["outputfolder"] = "outputs/" + hyperparameters["outfolder"]
hyperparameters["lbfgs_max_iterations"] = int(hyperparameters["lbfgs_max_iterations"])
hyperparameters["MassConservation"] = False


if not os.path.isdir(hyperparameters["outputfolder"]):
    os.makedirs(hyperparameters["outputfolder"], exist_ok=True)



(mesh, Img, NumData) = read_image(hyperparameters, name="input")


h = CellDiameter(mesh)
h = float(assemble(h*dx))

hyperparameters["expected_distance_covered"] = 15 # max. 15 voxels
v_needed = hyperparameters["expected_distance_covered"] / 1 
hyperparameters["DeltaT"] = float(h) / v_needed #1e-3
print("calculated initial time step size to", hyperparameters["DeltaT"])
hyperparameters["DeltaT_init"] = hyperparameters["DeltaT"]

with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)

(mesh_goal, Img_goal, NumData_goal) = read_image(hyperparameters, name="target", mesh=mesh)

#  breakpoint()

# output file
# fCont = XDMFFile(MPI.comm_world, hyperparameters["outputfolder"] + "/Control.xdmf")
# fCont .write(mesh, '/mesh')
# fCont.parameters["flush_output"] = True
# fCont.parameters["rewrite_function_mesh"] = False

controlFile = HDF5File(mesh.mpi_comm(), hyperparameters["outputfolder"] + "/Control.hdf", "w")
controlFile.write(mesh, "mesh")

stateFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/State.hdf", "w")
stateFile.write(mesh, "mesh")
# stateFile.parameters["flush_output"] = True
# stateFile.parameters["rewrite_function_mesh"] = False
# FOut.parameters["functions_share_mesh"] = True

velocityFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/VelocityField.hdf", "w")
velocityFile.write(mesh, "mesh")
# velocityFile.parameters["flush_output"] = True
# # velocityFile.parameters["rewrite_function_mesh"] = False

files = {
    "velocityFile": velocityFile,
    "stateFile": stateFile,
    "controlFile":controlFile
}

# transform colored image to black-white intensity image
Space = FunctionSpace(mesh, "DG", 1)
Img = project(Img, Space)
Img.rename("img", "")
Img_goal = project(Img_goal, Space)
NumData = 1

print("Projected data")

# initialize trafo
vCG = VectorFunctionSpace(mesh, "CG", 1)


if hyperparameters["smoothen"]:
    M_lumped = get_lumped_mass_matrix(vCG=vCG)
else:
    M_lumped = None

t0 = time.time()

for n in range(4):
    
    try:
        find_velocity(Img, Img_goal, vCG, M_lumped, hyperparameters, files)
    except CFLerror:
        hyperparameters["DeltaT"] *= 1 / 2
        print("CFL condition violated, reducing time step size and retry")
        pass


tcomp = (time.time()-t0) / 3600

hyperparameters["optimization_time_hours"] = tcomp

with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)
