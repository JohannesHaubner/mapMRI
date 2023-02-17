from fenics import *
from fenics_adjoint import *
import os
import json
import time
import argparse
import numpy as np
import nibabel
import resource

PETScOptions.set("mat_mumps_icntl_4", 3)   # verbosity

# set_log_level(LogLevel.CRITICAL)

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass


# raise ValueError

comm = MPI.comm_world
nprocs = comm.Get_size()


print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

import dgregister.config as config
config.hyperparameters = {"optimize": True}

from dgregister.helpers import get_lumped_mass_matrices
from dgregister.MRI2FEM import read_image, fem2mri

parser = argparse.ArgumentParser()

parser.add_argument("--outfoldername", type=str, default=None, help=""" name of folder to store to under "path + "output_dir" """)
parser.add_argument("--code_dir", type=str, default="/home/bastian/Oscar-Image-Registration-via-Transport-Equation/")
parser.add_argument("--logfile", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--slurmid", type=str, required=True)
parser.add_argument("--solver", default="krylov", choices=["lu", "krylov", "cg"])
parser.add_argument("--timestepping", default="RungeKutta", choices=["RungeKutta", "CrankNicolson", "explicitEuler"])
parser.add_argument("--smoothen", default=True, action="store_true", help="Obsolete flag. Use proper scalar product")
parser.add_argument("--nosmoothen", default=False, action="store_true", help="Sets smoothen=False")
parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--lbfgs_max_iterations", type=float, default=400)
parser.add_argument("--max_timesteps", type=float, default=None)
parser.add_argument("--state_functiondegree", type=int, default=1)

parser.add_argument("--not_store_solver", default=False, action="store_true", help="Do not store solve in preconditioning")
parser.add_argument("--preconditioning", default="preconditioning", type=str, choices=["none", "preconditioning"])
parser.add_argument("--memdebug", default=False, action="store_true")
parser.add_argument("--mpiwait", default=False, action="store_true", help="use MPI barrier in callback for debuggin")
parser.add_argument("--maxcor", default=10, type=int)
parser.add_argument("--padding", default=None, type=int)

parser.add_argument("--reassign", default=False, action="store_true")

parser.add_argument("--projector", default=False, action="store_true")
parser.add_argument("--tukey", default=False, action="store_true", help="Use tukey loss function")
parser.add_argument("--tukey_c", type=int, default=1)
parser.add_argument("--huber", default=False, action="store_true", help="Use Huber loss function")
parser.add_argument("--huber_delta", type=int, default=1)
parser.add_argument("--normalization_scale", type=float, default=1, help="divide both images with this number")
parser.add_argument("--equalmean", default=False, action="store_true", help="Scale both images to the same mean.")
parser.add_argument("--readname", type=str, default="-1")
parser.add_argument("--starting_guess", type=str, default=None)
parser.add_argument("--starting_state", type=str, default=None)
# parser.add_argument("--normalization", type=str, default="max")
# parser.add_argument("--multigrid", default=False, action="store_true", help="Use starting guess & another transform")

parser.add_argument("--filter", default=False, action="store_true", help="median filter on input and output")
parser.add_argument("--debug", default=False, action="store_true", help="Debug")
parser.add_argument("--timing", default=False, action="store_true")
parser.add_argument("--ocd", default=False, action="store_true")
parser.add_argument("--input", type=str)
parser.add_argument("--target", type=str)

hyperparameters = vars(parser.parse_args())
hyperparameters["interpolate"] = False

os.chdir(hyperparameters["code_dir"])
print_overloaded("Setting pwd to", hyperparameters["code_dir"])


if not hyperparameters["output_dir"].endswith("/"):
    hyperparameters["output_dir"] += "/"

suffix = ""


if (hyperparameters["outfoldername"] is not None) or len(hyperparameters["outfoldername"]) == 0:
    # Workaround since None is not interpreted as None by argparse
    if hyperparameters["outfoldername"].lower() != "none":

        suffix = hyperparameters["outfoldername"]
        
        if len(hyperparameters["outfoldername"]) > 0:
            assert "E" != hyperparameters["outfoldername"][0]
            assert "/" not in hyperparameters["outfoldername"]
            assert "LBFGS" not in hyperparameters["outfoldername"]
            assert "RK" not in hyperparameters["outfoldername"]

hyperparameters["outfoldername"] = ""

if hyperparameters["ocd"]:
    hyperparameters["outfoldername"] = "OCD"
elif hyperparameters["timestepping"] == "RungeKutta":
    hyperparameters["outfoldername"] = "RK"
elif hyperparameters["timestepping"] == "CrankNicolson":
    hyperparameters["outfoldername"] = "CN"
elif hyperparameters["timestepping"] == "explicitEuler":
    hyperparameters["outfoldername"] = "E"


if not hyperparameters["ocd"]:
    hyperparameters["outfoldername"] += str(int(hyperparameters["max_timesteps"]))

hyperparameters["outfoldername"] += "A" + str(hyperparameters["alpha"])
hyperparameters["outfoldername"] += "LBFGS" + str(int(hyperparameters["lbfgs_max_iterations"]))

if hyperparameters["nosmoothen"]:
    hyperparameters["outfoldername"] += "NOSMOOTHEN"

if hyperparameters["state_functiondegree"] == 0:
    hyperparameters["outfoldername"] += "DG0"

if hyperparameters["tukey"]:
    hyperparameters["outfoldername"] += "C" + str(hyperparameters["tukey_c"])

hyperparameters["outfoldername"] += suffix

print_overloaded("Generated outfoldername", hyperparameters["outfoldername"])

if hyperparameters["starting_state"] is not None:
    assert os.path.isfile(hyperparameters["starting_state"])

hyperparameters["normalize"] = False

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

if hyperparameters["ocd"]:
    from dgregister.find_velocity_ocd import find_velocity
else:
    from dgregister.find_velocity import find_velocity

print_overloaded("Setting config.hyperparameters")

for key, item in hyperparameters.items():
    print_overloaded(key, ":", item)

if not os.path.isdir(hyperparameters["outputfolder"]):
    os.makedirs(hyperparameters["outputfolder"], exist_ok=True)

state_functionspace=hyperparameters["state_functionspace"]
state_functiondegree=hyperparameters["state_functiondegree"] 


if hyperparameters["equalmean"]:
    iscale = np.mean(nibabel.load(hyperparameters["target"]).get_fdata()) / np.mean(nibabel.load(hyperparameters["input"]).get_fdata())
else:
    iscale = 1

hyperparameters["iscale"] = iscale


mem = resource.getrusage(resource.RUSAGE_SELF)[2]
print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", "-10", "process", str(MPI.rank(MPI.comm_world)))




(domainmesh, Img, input_max, projector) = read_image(filename=hyperparameters["input"], name="input", mesh=None, 
            iscale=iscale, hyperparameters=hyperparameters, normalization_scale=hyperparameters["normalization_scale"],
            state_functionspace=state_functionspace, state_functiondegree=state_functiondegree)

mem = resource.getrusage(resource.RUSAGE_SELF)[2]
print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", "-9", "process", str(MPI.rank(MPI.comm_world)))

vCG = VectorFunctionSpace(domainmesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"])


if hyperparameters["starting_guess"] is not None:

    assert "CurrentV.hdf" not in hyperparameters["starting_guess"]
    assert "Velocity" not in hyperparameters["starting_guess"]

    starting_guess = Function(vCG)

    hdf = HDF5File(domainmesh.mpi_comm(), hyperparameters["starting_guess"], "r")
    hdf.read(starting_guess, hyperparameters["readname"])
    hdf.close()

else:
    starting_guess = None

if hyperparameters["starting_state"] is not None:

    # assert hyperparameters["starting_state"] is not None
    # assert "CurrentV.hdf" not in hyperparameters["starting_guess"]
    # assert "Velocity" not in hyperparameters["starting_guess"]

    # controlfun = Function(vCG)

    # hdf = HDF5File(domainmesh.mpi_comm(), hyperparameters["starting_guess"], "r")
    # hdf.read(controlfun, hyperparameters["readname"])
    # hdf.close()
    
    Img = Function(FunctionSpace(domainmesh, "DG", 1))

    with XDMFFile(hyperparameters["starting_state"]) as xdmf:
        xdmf.read_checkpoint(Img, "CurrentState")

    print_overloaded("Loaded ", hyperparameters["starting_state"], "as starting guess for state")
else:
    controlfun = None

(mesh_goal, Img_goal, target_max, _) = read_image(hyperparameters["target"], name="target", mesh=domainmesh, projector=projector,
    iscale=iscale, hyperparameters=hyperparameters,normalization_scale=hyperparameters["normalization_scale"],
        state_functionspace=state_functionspace, state_functiondegree=state_functiondegree,)

mem = resource.getrusage(resource.RUSAGE_SELF)[2]
print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", "-8", "process", str(MPI.rank(MPI.comm_world)))


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

if not hyperparameters["memdebug"]:

    controlFile = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Control.hdf", "w")
    controlFile.write(domainmesh, "mesh")

    stateFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/State.hdf", "w")
    stateFile.write(domainmesh, "mesh")

    velocityFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/VelocityField.hdf", "w")
    velocityFile.write(domainmesh, "mesh")

    with XDMFFile(hyperparameters["outputfolder"] + "/Target.xdmf") as xdmf:
        xdmf.write_checkpoint(Img_goal, "Img_goal", 0.)

    with XDMFFile(hyperparameters["outputfolder"] + "/Input.xdmf") as xdmf:
        xdmf.write_checkpoint(Img, "Img", 0.)


    files["velocityFile"] = velocityFile
    files["stateFile"] = stateFile
    files["controlFile"] = controlFile


Img.rename("input", "")
Img_goal.rename("target", "")

if hyperparameters["smoothen"]:
    _, M_lumped_inv = get_lumped_mass_matrices(vCG=vCG)
else:
    M_lumped_inv = None

t0 = time.time()

files["lossfile"] = hyperparameters["outputfolder"] + '/loss.txt'
files["l2lossfile"] = hyperparameters["outputfolder"] + '/l2loss.txt'


mem = resource.getrusage(resource.RUSAGE_SELF)[2]
print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", "-7", "process", str(MPI.rank(MPI.comm_world)))

#####################################################################
# Optimization

return_values = find_velocity(starting_image=Img, Img_goal=Img_goal, vCG=vCG, M_lumped_inv=M_lumped_inv, 
    hyperparameters=hyperparameters, files=files, starting_guess=starting_guess)

print(return_values, len(return_values))
if not len(return_values) == 3:

    return_values = return_values[0]
    # print("return_values:", type(return_values))
    # print("return_values:", len(return_values))
    # print("return_values:", type(return_values[0]))
    # print("return_values:", len(return_values[0]))
    print("Len of return values is NOT 3")

else:
    
    print("Len of return values is 3")


FinalImg, FinalVelocity, FinalControl  = return_values[0], return_values[1], return_values[2]

tcomp = (time.time()-t0) / 3600
print_overloaded("Done with optimization, took", format(tcomp, ".1f"), "hours")

hyperparameters["optimization_time_hours"] = tcomp
hyperparameters["Jd_final"] = hyperparameters["Jd_current"]
hyperparameters["Jl2_final"] = hyperparameters["Jl2_current"]
if MPI.rank(MPI.comm_world) == 0:
    with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile, sort_keys=True, indent=4)

if hyperparameters["timing"] or hyperparameters["memdebug"]:
    exit()

#####################################################################



with XDMFFile(hyperparameters["outputfolder"] + "/Finalstate.xdmf") as xdmf:
    xdmf.write_checkpoint(FinalImg, "Finalstate", 0.)

with XDMFFile(hyperparameters["outputfolder"] + "/Finalvelocity.xdmf") as xdmf:
    xdmf.write_checkpoint(FinalVelocity, "FinalV", 0.)

files["velocityFile"].write(FinalVelocity, "-1")
files["controlFile"].write(FinalControl, "-1")
files["stateFile"].write(FinalImg, "-1")

print_overloaded("Stored final State, Control, Velocity to .hdf files")

try:
    if min(hyperparameters["input.shape"]) > 1 and len(hyperparameters["input.shape"]) == 3:
        
        retimage = fem2mri(function=FinalImg, shape=hyperparameters["input.shape"])
        if MPI.comm_world.rank == 0:
            nifti = nibabel.Nifti1Image(retimage, nibabel.load(hyperparameters["input"]).affine)

            nibabel.save(nifti, hyperparameters["outputfolder"] + '/Finalstate.mgz')

            print_overloaded("Stored mgz image of transformed image")
except:
    print("Something went wrong when storing final image to mgz")
    pass



else:
    pass

if hyperparameters["logfile"] is not None:
    print_overloaded("Trying to copy logfile")
    if MPI.rank(MPI.comm_world) == 0:
        os.system("cp -v " + hyperparameters["logfile"] + " " + hyperparameters["outputfolder"] + "/")
    
print_overloaded("Optimize3d.py ran succesfully :-)")