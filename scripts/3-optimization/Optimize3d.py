from fenics import *
from fenics_adjoint import *
import os
import json
import time
import argparse
import numpy as np


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass

print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

from dgregister.helpers import get_lumped_mass_matrices
from dgregister.MRI2FEM import read_image

parser = argparse.ArgumentParser()

# I/O
parser.add_argument("--logfile", type=str, default=None, help="path to log file if this should be stored")
parser.add_argument("--output_dir", type=str, default=None, help="path where output is stored")
parser.add_argument("--slurmid", type=str, required=True)
parser.add_argument("--input", type=str, help="input image as mgz")
parser.add_argument("--target", type=str, help="target image as mgz")
parser.add_argument("--smoothen_image", default=False, action="store_true", help="Apply Gauss filter")
# Starting guesses
parser.add_argument("--readname", type=str, default="-1", help="Name of the FEniCS function for --starting_guess")
parser.add_argument("--starting_guess", type=str, default=None, help="Initialize the optimization with this control")
parser.add_argument("--starting_state", type=str, default=None, help="Start with this image, given as a FEniCS function")
parser.add_argument("--statename", type=str, default="CurrentState",  help="Name of the FEniCS function for --starting_state")


# Forward pass 
parser.add_argument("--timestepping", default="RungeKutta", choices=["RungeKutta", "CrankNicolson", "explicitEuler"])
parser.add_argument("--max_timesteps", type=float, default=100)
parser.add_argument("--forward", default=False, action="store_true", help="Only transport an image forward.")

# Optimization
parser.add_argument("--alpha", type=float, default=1e-2)
parser.add_argument("--omega", type=float, default=0)
parser.add_argument("--epsilon", type=float, default=1)
parser.add_argument("--lbfgs_max_iterations", type=float, default=200)
parser.add_argument("--taylortest", default=False, action="store_true", help="Taylor test")

# Losses
# parser.add_argument("--huber", default=False, action="store_true", help="Use Huber loss function instead of L2")
parser.add_argument("--huber_delta", type=int, default=1)

hyperparameters = vars(parser.parse_args())

import dgregister.config as config

# Set the velocity smoothening parameters
config.EPSILON = hyperparameters["epsilon"]
config.OMEGA = hyperparameters["omega"]

from dgregister.find_velocity import find_velocity

if not hyperparameters["output_dir"].endswith("/"):
    hyperparameters["output_dir"] += "/"

hyperparameters["outfoldername"] = ""

if hyperparameters["timestepping"] == "RungeKutta":
    hyperparameters["outfoldername"] = "RK"
elif hyperparameters["timestepping"] == "CrankNicolson":
    hyperparameters["outfoldername"] = "CN"
elif hyperparameters["timestepping"] == "explicitEuler":
    hyperparameters["outfoldername"] = "E"

hyperparameters["outfoldername"] += format(hyperparameters["max_timesteps"], ".0f")
hyperparameters["outfoldername"] += "A" + str(hyperparameters["alpha"])
hyperparameters["outfoldername"] += "LBFGS" + str(int(hyperparameters["lbfgs_max_iterations"]))

print_overloaded("Generated outfoldername", hyperparameters["outfoldername"])

if hyperparameters["starting_state"] is not None:
    assert os.path.isfile(hyperparameters["starting_state"])

hyperparameters["normalize"] = False
hyperparameters["smoothen"] = True
hyperparameters["huber"] = True

hyperparameters["outputfolder"] = hyperparameters["output_dir"] + hyperparameters["outfoldername"]
hyperparameters["lbfgs_max_iterations"] = int(hyperparameters["lbfgs_max_iterations"])
hyperparameters["MassConservation"] = False
hyperparameters["velocity_functiondegree"] = 1
hyperparameters["velocity_functionspace"] = "CG"
hyperparameters["state_functionspace"] = "DG"
hyperparameters["state_functiondegree"] = 1

    
for key, item in hyperparameters.items():
    print_overloaded(key, ":", item)

if not os.path.isdir(hyperparameters["outputfolder"]):
    os.makedirs(hyperparameters["outputfolder"], exist_ok=True)


(domainmesh, Img, input_max, projector) = read_image(filename=hyperparameters["input"], name="input", mesh=None, hyperparameters=hyperparameters,
            state_functionspace=hyperparameters["state_functionspace"], state_functiondegree=hyperparameters["state_functiondegree"])

vCG = VectorFunctionSpace(domainmesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"])

if hyperparameters["starting_guess"] is not None:

    """Start the optimization with a velocity field given as FEniCS function
    """
    if not hyperparameters["forward"]:
        assert hyperparameters["starting_state"] is None
    assert "CurrentV.hdf" not in hyperparameters["starting_guess"]
    assert "Velocity" not in hyperparameters["starting_guess"]

    starting_guess = Function(vCG)

    try:
        with XDMFFile(hyperparameters["starting_guess"]) as xdmf:
            xdmf.read_checkpoint(starting_guess, "CurrentV")
        
    except:
        hdf = HDF5File(domainmesh.mpi_comm(), hyperparameters["starting_guess"].replace("Control_checkpoint.xdmf", "CurrentControl.hdf"), "r")
        hdf.read(starting_guess, "function")
        hdf.close()

    hfile = hyperparameters["starting_guess"].replace("Control_checkpoint.xdmf", "hyperparameters.json")

    assert os.path.isfile(hfile)

    old_hypers = json.load(open(hfile))
    
    if hyperparameters["forward"]:
        assert hyperparameters["max_timesteps"] == old_hypers["max_timesteps"]
        hyperparameters["lbfgs_max_iterations"] = 0

    if not hyperparameters["forward"]:        
        if old_hypers["starting_state"] is not None:

            hyperparameters["starting_state"] = old_hypers["starting_state"]

            print_overloaded("Will read starting state")
            print_overloaded(old_hypers["starting_state"])
            print_overloaded(" from previous velocity in order to restart L-BFGS-B")

            assert os.path.isfile(old_hypers["starting_state"])

    print_overloaded("Read starting guess")

    print_overloaded("norm", norm(starting_guess))

else:
    starting_guess = None

if hyperparameters["starting_state"] is not None:
    assert os.path.isfile(hyperparameters["starting_state"])
    Img = Function(FunctionSpace(domainmesh, "DG", 1))
    print_overloaded("Trying to read", hyperparameters["statename"], "from", hyperparameters["starting_state"])
    with XDMFFile(hyperparameters["starting_state"]) as xdmf:
        xdmf.read_checkpoint(Img, hyperparameters["statename"])

    print_overloaded("Loaded ", hyperparameters["starting_state"], "as starting guess for state")


(mesh_goal, Img_goal, target_max, _) = read_image(hyperparameters["target"], name="target", mesh=domainmesh, projector=projector,
        hyperparameters=hyperparameters, state_functionspace=hyperparameters["state_functionspace"], state_functiondegree=hyperparameters["state_functiondegree"])


hyperparameters["max_voxel_intensity"] = max(input_max, target_max)

print_overloaded("check:norms:", assemble(Img*dx(domainmesh)), assemble(Img_goal*dx(domainmesh)))

T_final = 1

h = CellDiameter(domainmesh)
h = float(assemble(h*dx))

hyperparameters["mehsh"] = h
hyperparameters["maxMeshCoordinate"] = np.max(domainmesh.coordinates())
hyperparameters["max_timesteps"] = int(hyperparameters["max_timesteps"])
hyperparameters["DeltaT"] = T_final / hyperparameters["max_timesteps"]

if MPI.rank(MPI.comm_world) == 0:
    with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile, sort_keys=True, indent=4)
else:
    pass

files = {}

with XDMFFile(hyperparameters["outputfolder"] + "/Target.xdmf") as xdmf:
    xdmf.write_checkpoint(Img_goal, "Img_goal", 0.)

with XDMFFile(hyperparameters["outputfolder"] + "/Input.xdmf") as xdmf:
    xdmf.write_checkpoint(Img, "Img", 0.)

Img.rename("input", "")
Img_goal.rename("target", "")

if hyperparameters["smoothen"]:
    _, M_lumped_inv = get_lumped_mass_matrices(vCG=vCG)
else:
    M_lumped_inv = None

t0 = time.time()

files["lossfile"] = hyperparameters["outputfolder"] + '/loss.txt'
files["l2lossfile"] = hyperparameters["outputfolder"] + '/l2loss.txt'


#####################################################################
# Optimization

# find_velocity(starting_image, Img_goal, vCG, M_lumped_inv, hyperparameters, files, starting_guess=None)
return_values = find_velocity(starting_image=Img, Img_goal=Img_goal, vCG=vCG, M_lumped_inv=M_lumped_inv, 
    hyperparameters=hyperparameters, files=files, starting_guess=starting_guess, storage_info=None)



if not len(return_values) == 3:
    # TODO FIXME
    # Something weird was happening here. MPI-related?
    print(return_values, len(return_values))
    return_values = return_values[0]
    print("Len of return values is NOT 3")
else:
    pass

FinalImg, FinalVelocity, FinalControl  = return_values[0], return_values[1], return_values[2]

tcomp = (time.time()-t0) / 3600
print_overloaded("Done with optimization, took", format(tcomp, ".1f"), "hours")


hyperparameters["optimization_time_hours"] = tcomp

if not hyperparameters["forward"]:
    hyperparameters["Jd_final"] = hyperparameters["Jd_current"]
    hyperparameters["Jl2_final"] = hyperparameters["Jl2_current"]

if MPI.rank(MPI.comm_world) == 0:
    with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile, sort_keys=True, indent=4)


#####################################################################


with XDMFFile(hyperparameters["outputfolder"] + "/Finalstate.xdmf") as xdmf:
    xdmf.write_checkpoint(FinalImg, "Finalstate", 0.)

with XDMFFile(hyperparameters["outputfolder"] + "/Finalvelocity.xdmf") as xdmf:
    xdmf.write_checkpoint(FinalVelocity, "FinalV", 0.)

controlFile = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Control.hdf", "w")
controlFile.write(domainmesh, "mesh")

stateFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/State.hdf", "w")
stateFile.write(domainmesh, "mesh")

velocityFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/VelocityField.hdf", "w")
velocityFile.write(domainmesh, "mesh")
files["velocityFile"] = velocityFile
files["stateFile"] = stateFile
files["controlFile"] = controlFile

files["velocityFile"].write(FinalVelocity, "-1")
files["controlFile"].write(FinalControl, "-1")
files["stateFile"].write(FinalImg, "-1")

print_overloaded("Stored final State, Control, Velocity to .hdf files")

if hyperparameters["logfile"] is not None:
    print_overloaded("Trying to copy logfile")
    if MPI.rank(MPI.comm_world) == 0:
        os.system("cp -v " + hyperparameters["logfile"] + " " + hyperparameters["outputfolder"] + "/")
    
print_overloaded("Optimize3d.py ran succesfully :-)")