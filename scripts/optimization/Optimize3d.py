from fenics import *
from fenics_adjoint import *
import os
import json
import time
import argparse
import numpy as np
import nibabel
import pathlib


PETScOptions.set("mat_mumps_icntl_4", 3)   # verbosity

# set_log_level(LogLevel.CRITICAL)

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass

print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

from dgregister.helpers import get_lumped_mass_matrices
from dgregister.MRI2FEM import read_image, fem2mri
from dgregister.find_velocity import find_velocity
from dgregister.MRI2FEM import fem2mri
from dgregister.helpers import crop_to_original

parser = argparse.ArgumentParser()

# I/O
parser.add_argument("--logfile", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--slurmid", type=str, required=True)
parser.add_argument("--input", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--memdebug", default=False, action="store_true", help="Taylor test")
# Starting guesses
parser.add_argument("--readname", type=str, default="-1")
parser.add_argument("--starting_guess", type=str, default=None)
parser.add_argument("--starting_state", type=str, default=None)

# Forward pass 
parser.add_argument("--timestepping", default="RungeKutta", choices=["RungeKutta", "CrankNicolson", "explicitEuler"])
parser.add_argument("--max_timesteps", type=float, default=None)
parser.add_argument("--storeto", type=str, help="Folder with results.Store npy and mgz files to folder during forward pass")

# Optimization
parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--lbfgs_max_iterations", type=float, default=400)
parser.add_argument("--maxcor", default=10, type=int)
parser.add_argument("--taylortest", default=False, action="store_true", help="Taylor test")

# Losses
parser.add_argument("--tukey", default=False, action="store_true", help="Use tukey loss function")
parser.add_argument("--tukey_c", type=int, default=1)
parser.add_argument("--huber", default=False, action="store_true", help="Use Huber loss function")
parser.add_argument("--huber_delta", type=int, default=1)

hyperparameters = vars(parser.parse_args())

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





if hyperparameters["storeto"] is not None:

    if MPI.rank(MPI.comm_world) > 0:
        raise NotImplementedError

    original_hyperparameters = json.load(open(pathlib.Path(hyperparameters["storeto"]) / "hyperparameters.json"))

    assert hyperparameters["max_timesteps"] == original_hyperparameters["max_timesteps"]

    # old path: "/home/bastian/D1/registration/mri2fem-dataset/hydrocephalus/068_brain_nyul_ventriclemasked.mgz"
    # new path: /home/bastian/D1/registration/hydrocephalus/ventricles/021to068_nyul_ventriclemasked.mgz

    if not os.path.isfile(original_hyperparameters["input"]):

        assert hyperparameters["input"] == original_hyperparameters["input"].replace("mri2fem-dataset/hydrocephalus", "hydrocephalus/ventricles")
        assert hyperparameters["target"] == original_hyperparameters["target"].replace("mri2fem-dataset/hydrocephalus", "hydrocephalus/ventricles")

    assert hyperparameters["timestepping"] == original_hyperparameters["timestepping"]

    if hyperparameters["starting_guess"] is not None or hyperparameters["starting_state"] is not None:
        raise NotImplementedError
    
    MPI.barrier(MPI.comm_world)

    hyperparameters["starting_guess"] = str(pathlib.Path(hyperparameters["storeto"]) / "Control_checkpoint.xdmf")
    assert os.path.isfile(hyperparameters["starting_guess"])
    hyperparameters["readname"] = "CurrentV"
    # Control_checkpoint.xdmf --readname CurrentV \

    if "ventricle" in hyperparameters["input"] or "hydrocephalus" in hyperparameters["input"]:

        box = np.load("/home/bastian/D1/registration/hydrocephalus/freesurfer/021/testouts/box_all.npy")
        space = 2
        pad = 2

        aff3 = nibabel.load("/home/bastian/D1/registration/hydrocephalus/normalized/registered/021to068.mgz").affine

    else:
        assert "abby" in hyperparameters["input"]
        assert "ernie" in hyperparameters["target"]
        box = np.load("/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/box.npy")
        space = 0
        pad = 2

        aff3 = nibabel.load("/home/bastian/D1/registration/mri2fem-dataset/normalized/registered/abbytoernie.mgz").affine

    
    storage_info = {"shape": hyperparameters["input.shape"], 
                   "store_states_to": hyperparameters["storeto"], 
                   "pad": pad, "space": space, 
                   "box": box, "aff": aff3
                   }
    
    # box, aff = np.ndarray

    store_states_to = pathlib.Path(hyperparameters["storeto"]) / "states"

    storage_info["store_states_to"] = str(store_states_to)

    os.makedirs(store_states_to, exist_ok=True)

    MPI.barrier(MPI.comm_world)

    assert os.path.isdir(store_states_to)

else:
    storage_info = None



if hyperparameters["storeto"] is not None:

    retimage = fem2mri(function=Img, shape=storage_info["shape"])

    MPI.barrier(MPI.comm_world)

    print(storage_info)

    storepath = store_states_to / ("state" + format(0, ".0f") + ".npy")

    np.save(storepath, retimage)

    filled_image = crop_to_original(orig_image=np.zeros((256, 256, 256)), 
                                    cropped_image=retimage, box=storage_info["box"], 
                                    space=storage_info["space"], pad=storage_info["pad"])

    nii = nibabel.Nifti1Image(filled_image, storage_info["aff"])
    nibabel.save(nii, str(storepath).replace(".npy", ".mgz"))

    print_overloaded("-- stored", str(storepath))


vCG = VectorFunctionSpace(domainmesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"])

if hyperparameters["starting_guess"] is not None:

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

    if old_hypers["starting_state"] is not None:

        hyperparameters["starting_state"] = old_hypers["starting_state"]

        print_overloaded("Will read starting state")
        print_overloaded(old_hypers["starting_state"])
        print(" from previous velocity in order to restart L-BFGS-B")

        assert os.path.isfile(old_hypers["starting_state"])

    print_overloaded("Read starting guess")

    print_overloaded("norm", norm(starting_guess))

else:
    starting_guess = None

if hyperparameters["starting_state"] is not None:

    Img = Function(FunctionSpace(domainmesh, "DG", 1))

    with XDMFFile(hyperparameters["starting_state"]) as xdmf:
        xdmf.read_checkpoint(Img, "CurrentState")

    print_overloaded("Loaded ", hyperparameters["starting_state"], "as starting guess for state")
else:
    controlfun = None

(mesh_goal, Img_goal, target_max, _) = read_image(hyperparameters["target"], name="target", mesh=domainmesh, projector=projector,
        hyperparameters=hyperparameters, state_functionspace=hyperparameters["state_functionspace"], state_functiondegree=hyperparameters["state_functiondegree"])


if hyperparameters["storeto"] is not None:

    retimage = fem2mri(function=Img_goal, shape=storage_info["shape"])

    storepath = store_states_to / ("state" + format(10000, ".0f") + ".npy")

    np.save(storepath, retimage)

    filled_image = crop_to_original(orig_image=np.zeros((256, 256, 256)), 
                                    cropped_image=retimage, box=storage_info["box"], 
                                    space=storage_info["space"], pad=storage_info["pad"])

    nii = nibabel.Nifti1Image(filled_image, storage_info["aff"])
    nibabel.save(nii, str(storepath).replace(".npy", ".mgz"))

    print_overloaded("-- stored", str(storepath))

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
    hyperparameters=hyperparameters, files=files, starting_guess=starting_guess, storage_info=storage_info)

# TODO FIXME
# Something weird was happening here. 
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