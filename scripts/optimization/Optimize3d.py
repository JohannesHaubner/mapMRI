from fenics import *
from fenics_adjoint import *
import os
import json
import time
import argparse
import numpy as np
import nibabel

set_log_level(LogLevel.CRITICAL)

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass


comm = MPI.comm_world
nprocs = comm.Get_size()


print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

import dgregister.config as config
config.hyperparameters = {"optimize": True}

from dgregister.helpers import load_control, get_lumped_mass_matrices
from dgregister.MRI2FEM import read_image, fem2mri

parser = argparse.ArgumentParser()

parser.add_argument("--outfoldername", type=str, default=None, help=""" name of folder to store to under "path + "output_dir" """)
parser.add_argument("--code_dir", type=str, default="/home/bastian/Oscar-Image-Registration-via-Transport-Equation/")
parser.add_argument("--logfile", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--slurmid", type=str, required=True)
parser.add_argument("--solver", default="krylov", choices=["lu", "krylov"])
parser.add_argument("--timestepping", default="RungeKutta", choices=["RungeKutta", "CrankNicolson", "explicitEuler"])
parser.add_argument("--smoothen", default=True, action="store_true", help="Obsolete flag. Use proper scalar product")
parser.add_argument("--nosmoothen", default=False, action="store_true", help="Sets smoothen=False")
parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--lbfgs_max_iterations", type=float, default=400)
parser.add_argument("--max_timesteps", type=float, default=None)
parser.add_argument("--state_functiondegree", type=int, default=1)


parser.add_argument("--preconditioning", required=True, type=str, choices=["none", "preconditioning"])
parser.add_argument("--memdebug", default=False, action="store_true")
parser.add_argument("--maxcor", default=10, type=int)

parser.add_argument("--projector", default=False, action="store_true")
parser.add_argument("--tukey", default=False, action="store_true", help="Use tukey loss function")
parser.add_argument("--tukey_c", type=int, default=4)
parser.add_argument("--normalization_scale", type=float, default=255, help="divide both images with this number")
parser.add_argument("--readname", type=str, default="-1")
parser.add_argument("--starting_guess", type=str, default=None)
# parser.add_argument("--normalization", type=str, default="max")
parser.add_argument("--multigrid", default=False, action="store_true", help="Use starting guess & another transform")

parser.add_argument("--filter", default=False, action="store_true", help="median filter on input and output")
parser.add_argument("--debug", default=False, action="store_true", help="Debug")
parser.add_argument("--timing", default=False, action="store_true")
parser.add_argument("--ocd", default=False, action="store_true")
parser.add_argument("--input", default="mridata_3d/091registeredto205_padded_coarsened.mgz")
parser.add_argument("--target", default="mridata_3d/205_cropped_padded_coarsened.mgz")

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

if hyperparameters["starting_guess"] is not None:
    assert os.path.isfile(hyperparameters["starting_guess"])

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

iscale = np.mean(nibabel.load(hyperparameters["target"]).get_fdata()) / np.mean(nibabel.load(hyperparameters["input"]).get_fdata())
hyperparameters["iscale"] = iscale


(domainmesh, Img, input_max) = read_image(filename=hyperparameters["input"], name="input", mesh=None, 
            iscale=iscale, hyperparameters=hyperparameters, normalization_scale=hyperparameters["normalization_scale"],
            state_functionspace=state_functionspace, state_functiondegree=state_functiondegree)

vCG = VectorFunctionSpace(domainmesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"])


if hyperparameters["starting_guess"] is not None:
    if hyperparameters["multigrid"] and hyperparameters["smoothen"]:
        assert "control" not in hyperparameters["starting_guess"].lower()
    else:
        assert "CurrentV.hdf" not in hyperparameters["starting_guess"]
        assert "Velocity" not in hyperparameters["starting_guess"]

    controlfun = load_control(hyperparameters, vCG)


else:
    controlfun = None

(mesh_goal, Img_goal, target_max) = read_image(hyperparameters["target"], name="target", mesh=domainmesh, 
    iscale=iscale, hyperparameters=hyperparameters,normalization_scale=hyperparameters["normalization_scale"],
        state_functionspace=state_functionspace, state_functiondegree=state_functiondegree,)




hyperparameters["max_voxel_intensity"] = max(input_max, target_max)

# # iscale = target_mean / input_mean
# iscale = np.mean(Img_goal.vector()[:]) / np.mean(Img.vector()[:])
# hyperparameters["iscale"] = iscale
# print_overloaded("Intensity scale factor =", iscale)

# Img.vector()[:] *= sqrt(iscale)
# Img_goal.vector()[:] /= sqrt(iscale)

print_overloaded("check:norms:", assemble(Img*dx(domainmesh)), assemble(Img_goal*dx(domainmesh)))

# assert np.allclose(assemble(Img*dx(domainmesh)), assemble(Img_goal*dx(domainmesh)), rtol=1e-1)




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

# print_overloaded("Img.vector()[:].mean()", Img.vector()[:].mean())
# print_overloaded("Img_goal.vector()[:].mean()", Img_goal.vector()[:].mean())

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

files = {
    "velocityFile": velocityFile,
    "stateFile": stateFile,
    "controlFile":controlFile
}

Img.rename("input", "")
Img_goal.rename("target", "")

if hyperparameters["smoothen"]:
    _, M_lumped_inv = get_lumped_mass_matrices(vCG=vCG)
else:
    M_lumped_inv = None

t0 = time.time()

files["lossfile"] = hyperparameters["outputfolder"] + '/loss.txt'
files["l2lossfile"] = hyperparameters["outputfolder"] + '/l2loss.txt'
files["regularizationfile"] = hyperparameters["outputfolder"] + '/regularization.txt'
files["totallossfile"] = hyperparameters["outputfolder"] + '/loss.txt'

# files["memoryfile"] = hyperparameters["outputfolder"] + '/memory.txt'


#####################################################################
# Optimization

FinalImg, FinalVelocity, FinalControl = find_velocity(Img=Img, Img_goal=Img_goal, vCG=vCG, M_lumped_inv=M_lumped_inv, 
    hyperparameters=hyperparameters, files=files, starting_guess=controlfun)

tcomp = (time.time()-t0) / 3600
print_overloaded("Done with optimization, took", format(tcomp, ".1f"), "hours")

hyperparameters["optimization_time_hours"] = tcomp
hyperparameters["Jd_final"] = hyperparameters["Jd_current"]
hyperparameters["Jreg_final"] = hyperparameters["Jreg_current"]
hyperparameters["Jl2_final"] = hyperparameters["Jl2_current"]
if MPI.rank(MPI.comm_world) == 0:
    with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile, sort_keys=True, indent=4)

if hyperparameters["timing"]:
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