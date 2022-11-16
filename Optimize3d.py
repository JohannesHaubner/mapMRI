from dolfin import *
from dolfin_adjoint import *
from DGTransport import Transport
#from SUPGTransport import Transport
# from Pic2Fen import *
import os
# from ipopt_solver import IPOPTProblem as IpoptProblem
# from ipopt_solver import IPOPTSolver as IpoptSolver
from mri_utils.MRI2FEM import read_image
import json
import time
import argparse
from transformation_overloaded import transformation
from preconditioning_overloaded import preconditioning
import numpy as np
import h5py

# PETScOptions.set("mat_mumps_use_omp_threads", 8)
# PETScOptions.set("mat_mumps_icntl_35", True) # set use of BLR (Block Low-Rank) feature (0:off, 1:optimal)
# PETScOptions.set("mat_mumps_cntl_7", 1e-8) # set BLR relaxation
# PETScOptions.set("mat_mumps_icntl_4", 3)   # verbosity
# PETScOptions.set("mat_mumps_icntl_24", 1)  # detect null pivot rows
# PETScOptions.set("mat_mumps_icntl_22", 0)  # out of core
# #PETScOptions.set("mat_mumps_icntl_14", 250) # max memory increase in %


parser = argparse.ArgumentParser()
parser.add_argument("--outfolder", required=True, type=str, help=""" name of folder to store to under "path + "outputs/" """)
parser.add_argument("--solver", default="lu", choices=["lu", "krylov"])
parser.add_argument("--timestepping", default="Crank-Nicolson", choices=["CrankNicolson", "explicitEuler"])
parser.add_argument("--smoothen", default=False, action="store_true", help="Use proper scalar product")
parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--starting_guess", type=str, default=None)

hyperparameters = vars(parser.parse_args())

for key, item in hyperparameters.items():
    print(key, ":", item)

assert "/" not in hyperparameters["outfolder"]

set_log_level(20)

if "home/bastian/" in os.getcwd():
    path = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/"


hyperparameters["Input"] = path + "mridata_3d/091registeredto205_padded_coarsened.mgz"
hyperparameters["Target"] = path + "mridata_3d/205_cropped_padded_coarsened.mgz"
hyperparameters["outputfolder"] = path + "outputs/" + hyperparameters["outfolder"] # "output_coarsened_mri_ipopt"
hyperparameters["lbfgs_max_iterations"] = 400
hyperparameters["DeltaT"] = 1e-3
hyperparameters["MaxIter"] = 50
hyperparameters["MassConservation"] = False


if not os.path.isdir(hyperparameters["outputfolder"]):
    os.makedirs(hyperparameters["outputfolder"], exist_ok=True)

with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)

(mesh, Img, NumData) = read_image(hyperparameters["Input"])




(mesh_goal, Img_goal, NumData_goal) = read_image(hyperparameters["Target"], mesh)

#  breakpoint()

# output file
fCont = XDMFFile(MPI.comm_world, hyperparameters["outputfolder"] + "/Control.xdmf")
fCont.parameters["flush_output"] = True
fCont.parameters["rewrite_function_mesh"] = False

fState = XDMFFile(MPI.comm_world, hyperparameters["outputfolder"] + "/State.xdmf")
fState.parameters["flush_output"] = True
fState.parameters["rewrite_function_mesh"] = False
# output file
if hyperparameters["smoothen"]:
    fTrafo = XDMFFile(MPI.comm_world, hyperparameters["outputfolder"] + "/Trafo.xdmf")
    fTrafo.parameters["flush_output"] = True
    fTrafo.parameters["rewrite_function_mesh"] = False

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
    s1 = TrialFunction(vCG)
    s2 = TestFunction(vCG)
    form = inner(s1, s2) * dx
    mass_action_form = action(form, Constant((1., 1., 1.)))
    M_lumped = assemble(form)
    M_lumped_inv = assemble(form)
    M_lumped.zero()
    M_lumped_inv.zero()
    diag = assemble(mass_action_form)
    diag[:] = np.sqrt(diag[:])
    diaginv = assemble(mass_action_form)
    diaginv[:] = 1.0/np.sqrt(diag[:])
    M_lumped.set_diagonal(diag)
    M_lumped_inv.set_diagonal(diaginv)

set_working_tape(Tape())

# initialize control
controlfun = Function(vCG)

if hyperparameters["starting_guess"] is not None:

    print("Will try to read starting guess")
    assert os.path.isfile(hyperparameters["starting_guess"])

    h5file = h5py.File(hyperparameters["starting_guess"])

    # mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), hyperparameters["starting_guess"], "r")

    hdf.read(controlfun, "VisualisationVector")

    breakpoint()

if hyperparameters["smoothen"]:
    controlf = transformation(controlfun, M_lumped)
    control = preconditioning(controlf, smoothen=True)
else:
    control = controlfun

control.rename("control", "")

File(hyperparameters["outputfolder"] + "/input.pvd") << Img
File(hyperparameters["outputfolder"] + "/target.pvd") << Img_goal

# set_working_tape(Tape())
# #initialize control
# vCG = VectorFunctionSpace(mesh, "CG", 1)
# controlfun = Function(vCG)
# #x = SpatialCoordinate(mesh)
# #controlfun = project(as_vector((0.0, x[1])), vCG)

# control = preconditioning(controlfun, smoothen=hyperparameters["smoothen"])
# control.rename("control", "")

print("Running Transform()")

Img_deformed = Transport(Img, control, hyperparameters["MaxIter"], hyperparameters["DeltaT"], 
                            timestepping=hyperparameters["timestepping"], 
                           solver=hyperparameters["solver"], MassConservation=False)
#File( outputfolder + "/test.pvd") << Img_deformed

# solve forward and evaluate objective
alpha = Constant(hyperparameters["alpha"]) #regularization

state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
# cont = Control(controlfun)
# BZ: i think the lower is correct:
cont = Control(control)

J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx + alpha*grad(control)**2*dx(domain=mesh))

Jhat = ReducedFunctional(J, cont)

optimization_iterations = 0

# f = File( outputfolder + "/State_during_optim.pvd")

def cb(*args, **kwargs):
    global optimization_iterations
    optimization_iterations += 1
    current_pde_solution = state.tape_value()
    current_pde_solution.rename("Img", "")
    current_control = cont.tape_value()
    current_control.rename("control", "")

    # f << state

    fCont.write(current_control, float(optimization_iterations))
    fState.write(current_pde_solution, float(optimization_iterations))
    
    # FName =  outputfolder + "/optimize_%5d.png"%optimization_iterations
    # FEM2Pic(current_pde_solution, NumData, FName)
  
fState.write(Img_deformed, float(0))
fCont.write(control, float(0))
    
t0 = time.time()


optimization_iterations = 0

def cb(*args, **kwargs):
    global optimization_iterations
    optimization_iterations += 1
    
    current_pde_solution = state.tape_value()
    current_pde_solution.rename("Img", "")
    
    current_control = cont.tape_value()
    current_control.rename("control", "")
    
    if hyperparameters["smoothen"]:
        current_trafo = transformation(current_control, M_lumped)
        current_trafo = preconditioning(current_trafo, smoothen=True)
        current_trafo.rename("transformation", "")
        fTrafo.write(current_trafo, float(optimization_iterations))

    fCont.write(current_control, float(optimization_iterations))
    fState.write(current_pde_solution, float(optimization_iterations))
    
    # FName = "output" + filename + "/optimize_%5d.png"%optimization_iterations
    # FEM2Pic(current_pde_solution, NumData, FName)
  
fState.write(Img_deformed, float(0))
fCont.write(control, float(0))


minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)

# confun = Function(vCG)
# confun.vector().set_local(controlfun)
# File("output" + filename + "/OptControl.pvd") << confun


# minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)

tcomp = (time.time()-t0) / 3600

hyperparameters["optimization_time_hours"] = tcomp

with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)
