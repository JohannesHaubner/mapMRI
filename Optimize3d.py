from dolfin import *
from dolfin_adjoint import *
from DGTransport import Transport
import os
import json
import time
import argparse
import numpy as np
from transformation_overloaded import transformation
from preconditioning_overloaded import preconditioning
from mri_utils.helpers import load_velocity, get_lumped_mass_matrix
from mri_utils.MRI2FEM import read_image


parser = argparse.ArgumentParser()

parser.add_argument("--outfolder", required=True, type=str, help=""" name of folder to store to under "path + "outputs/" """)
parser.add_argument("--code_dir", type=str, default="/home/bastian/Oscar-Image-Registration-via-Transport-Equation/")
parser.add_argument("--solver", default="lu", choices=["lu", "krylov"])
parser.add_argument("--timestepping", default="Crank-Nicolson", choices=["CrankNicolson", "explicitEuler"])
parser.add_argument("--smoothen", default=False, action="store_true", help="Use proper scalar product")
parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--lbfgs_max_iterations", type=float, default=100)
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
hyperparameters["DeltaT"] = 1e-3
hyperparameters["MaxIter"] = 50
hyperparameters["MassConservation"] = False


if not os.path.isdir(hyperparameters["outputfolder"]):
    os.makedirs(hyperparameters["outputfolder"], exist_ok=True)

with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)

(mesh, Img, NumData) = read_image(hyperparameters, name="input")

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


velocityFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/VelocityField.hdf", "w")
velocityFile.write(mesh, "mesh")
# velocityFile.parameters["flush_output"] = True
# # velocityFile.parameters["rewrite_function_mesh"] = False

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

set_working_tape(Tape())

# initialize control
controlfun = Function(vCG)

if hyperparameters["starting_guess"] is not None:
    load_velocity(hyperparameters, controlfun=controlfun)


if hyperparameters["smoothen"]:
    controlf = transformation(controlfun, M_lumped)
else:
    controlf = controlfun

control = preconditioning(controlf, smoothen=hyperparameters["smoothen"])

control.rename("control", "")

File(hyperparameters["outputfolder"] + "/input.pvd") << Img
File(hyperparameters["outputfolder"] + "/target.pvd") << Img_goal

print("Running Transport()")

Img_deformed = Transport(Img, control, hyperparameters["MaxIter"], hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                           solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"])

# solve forward and evaluate objective
alpha = Constant(hyperparameters["alpha"]) #regularization

state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
cont = Control(controlfun)

J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx + alpha*grad(control)**2*dx(domain=mesh))

Jhat = ReducedFunctional(J, cont)

current_iteration = 0

stateFile.write(Img_deformed, str(current_iteration))

controlFile.write(control, str(current_iteration))

# controlFile.close()
# print("Wrote fCont, close and exit")
# exit()

print("Wrote fCont0")

t0 = time.time()

def cb(*args, **kwargs):
    global current_iteration
    current_iteration += 1
    
    current_pde_solution = state.tape_value()
    current_pde_solution.rename("Img", "")
    
    current_control = cont.tape_value()
    current_control.rename("control", "")
    
    if hyperparameters["smoothen"]:
        scaledControl = transformation(current_control, M_lumped)

    else:
        scaledControl = current_control

    velocityField = preconditioning(scaledControl, smoothen=hyperparameters["smoothen"])
    velocityField.rename("velocity", "")
    
    velocityFile.write(velocityField, str(current_iteration))
    controlFile.write(current_control, str(current_iteration))
    stateFile.write(current_pde_solution, str(current_iteration))

minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)

tcomp = (time.time()-t0) / 3600

hyperparameters["optimization_time_hours"] = tcomp

with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)
