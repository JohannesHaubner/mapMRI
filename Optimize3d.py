from dolfin import *
from dolfin_adjoint import *
from DGTransport import Transport
#from SUPGTransport import Transport
# from Pic2Fen import *
import os
from ipopt_solver import IPOPTProblem as IpoptProblem
from ipopt_solver import IPOPTSolver as IpoptSolver
from mri_utils.MRI2FEM import read_image
import json
import time
import argparse
from preconditioning_overloaded import preconditioning
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--outfolder", required=True, type=str, help=""" name of folder to store to under "path + "outputs/" """)
parser.add_argument("--use_krylov_solver", default=False, action="store_true")
parser.add_argument("--timestepping", default="Crank-Nicolson", choices=["CrankNicolson", "explicitEuler"])
parser.add_argument("--smoothen", default=False, action="store_true", help="Use proper scalar product")

hyperparameters = vars(parser.parse_args())

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
hyperparameters["alpha"] = 1e-4


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


# transform colored image to black-white intensity image
Space = FunctionSpace(mesh, "DG", 1)
Img = project(Img, Space)
Img.rename("img", "")
Img_goal = project(Img_goal, Space)
NumData = 1

File(hyperparameters["outputfolder"] + "/input.pvd") << Img
File(hyperparameters["outputfolder"] + "/target.pvd") << Img_goal

set_working_tape(Tape())

#initialize control
vCG = VectorFunctionSpace(mesh, "CG", 1)
controlfun = Function(vCG)
#x = SpatialCoordinate(mesh)
#controlfun = project(as_vector((0.0, x[1])), vCG)

control = preconditioning(controlfun, smoothen=hyperparameters["smoothen"])
control.rename("control", "")


Img_deformed = Transport(Img, control, hyperparameters["MaxIter"], hyperparameters["DeltaT"], 
                            timestepping=hyperparameters["timestepping"], 
                           use_krylov_solver=hyperparameters["use_krylov_solver"], MassConservation=False)
#File( outputfolder + "/test.pvd") << Img_deformed

# solve forward and evaluate objective
alpha = Constant(hyperparameters["alpha"]) #regularization

state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
cont = Control(controlfun)

if not hyperparameters["smoothen"]:
    J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx + alpha*grad(control)**2*dx(domain=mesh))
else:
    J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx)

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


if hyperparameters["smoothen"]:
    # IPOPT
    s1 = TrialFunction(vCG)
    s2 = TestFunction(vCG)
    form = inner(s1, s2) * dx
    mass_action_form = action(form, Constant((1., 1., 1.)))
    mass_action = assemble(mass_action_form)
    ndof = mass_action.size()
    diag_entries = mass_action.gather(range(ndof))
    problem = IpoptProblem([Jhat], [1.0], [], [], [], [], diag_entries, alpha.values()[0])
    ipopt = IpoptSolver(problem, callback = cb)
    controlfun = ipopt.solve(cont.vector()[:])
else:
    minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)

# confun = Function(vCG)
# confun.vector().set_local(controlfun)
# File("output" + filename + "/OptControl.pvd") << confun


# minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)

tcomp = (time.time()-t0) / 3600

hyperparameters["optimization_time_hours"] = tcomp

with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)
