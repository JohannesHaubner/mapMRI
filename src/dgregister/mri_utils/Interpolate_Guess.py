from fenics import *
from fenics_adjoint import *
import os
import argparse

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


from dgregister.helpers import load_control, interpolate_velocity

parser = argparse.ArgumentParser()

parser.add_argument("--outfoldername", required=True, type=str, help=""" name of folder to store to under "path + "output_dir" """)
parser.add_argument("--code_dir", type=str, default="/home/bastian/Oscar-Image-Registration-via-Transport-Equation/")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--readname", type=str, default="-1")
parser.add_argument("--function", type=str, default=None)
parser.add_argument("--debug", default=False, action="store_true", help="Debug")

hyperparameters = vars(parser.parse_args())



os.chdir(hyperparameters["code_dir"])
print_overloaded("Setting pwd to", hyperparameters["code_dir"])

assert "/" not in hyperparameters["outfoldername"]

if hyperparameters["function"] is not None:
    assert os.path.isfile(hyperparameters["function"])

set_log_level(20)

hyperparameters["outputfolder"] = hyperparameters["output_dir"] + hyperparameters["outfoldername"]
hyperparameters["functiondegree"] = 1
hyperparameters["velocity_functionspace"] = "CG"

for key, item in hyperparameters.items():
    print_overloaded(key, ":", item)

if not os.path.isdir(hyperparameters["outputfolder"]):
    os.makedirs(hyperparameters["outputfolder"], exist_ok=True)

hyperparameters["starting_guess"] = hyperparameters["function"]

domainmesh, vCG, controlfun = load_control(hyperparameters, controlfun=None)

domainmesh, vCG, controlfun = interpolate_velocity(hyperparameters, domainmesh, vCG, controlfun)

print_overloaded("Reading, interpolating and writing successful")