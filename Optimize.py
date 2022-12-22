from dolfin import *
from dolfin_adjoint import *

set_log_level(LogLevel.CRITICAL)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cube", type=str, choices=["true", "false"])
parser.add_argument("--newTransport",type=str, choices=["true", "false"])
parser.add_argument("--alpha", default=1e-4, type=float)
parser.add_argument("--regularize", choices=["L2control", "velocity"])
parser.add_argument("--loading", choices=["Pic2FEN", "MRI2FEM"])
parser.add_argument("--normalize", type=str, choices=["true", "false"])
parser.add_argument("--timestepping", default="RungeKutta", choices=["RungeKutta","RungeKuttaBug", "CrankNicolson", "explicitEuler"])


parser.add_argument("--maxiter", default=800, type=float)
parser.add_argument("--maxlbfs", default=1000, type=float)
hyperparameters= vars(parser.parse_args())

import config
hyperparameters["smoothen"] = True
hyperparameters["solver"] = "krylov"
hyperparameters["preconditioner"] = "amg"
hyperparameters["velocity_functiondegree"] = 1
hyperparameters["velocity_functionspace"] = "CG"
hyperparameters["state_functiondegree"] = 1
hyperparameters["state_functionspace"] = "DG"

config.hyperparameters = hyperparameters


from ipopt_solver import IPOPTProblem as IpoptProblem
from ipopt_solver import IPOPTSolver as IpoptSolver
#from SUPGTransport import Transport
from Pic2Fen import *
import json


from transformation_overloaded import transformation
from preconditioning_overloaded import preconditioning

import numpy

print("Using", hyperparameters["timestepping"])


for key in ["cube", "newTransport", "normalize"]:
    if hyperparameters[key] == "true":
        hyperparameters[key] = True
    else:
        hyperparameters[key] = False


set_log_level(20)

# read image
# filename = "mask_only"
filename = ""
if hyperparameters["cube"]:
    print("Using Cube as test images")
    FName = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d/input.mgz"
    FName_goal = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d/target.mgz"
else:
    print("Using rocket")
    FName = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/shuttle_small.png"
    FName_goal = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/shuttle_goal.png"





if hyperparameters["loading"] == "Pic2FEN":

    (mesh, Img, NumData) = Pic2FEM(FName)
    #FName_goal = "./braindata_2d/slice091" + filename +".png"
    (mesh_goal, Img_goal, NumData_goal) = Pic2FEM(FName_goal, mesh)

    # transform colored image to black-white intensity image
    Space = FunctionSpace(mesh, "DG", 1)
    Img = project(sqrt(inner(Img, Img)), Space)
    Img.rename("img", "")
    Img_goal = project(sqrt(inner(Img_goal, Img_goal)), Space)
    NumData = 1


    if hyperparameters["normalize"]:

        Img.vector()[:] *= 1 / Img.vector()[:].max()
        Img.vector()[:] = np.where(Img.vector()[:] < 0, 0, Img.vector()[:])
        Img_goal.vector()[:] *= 1 / Img_goal.vector()[:].max()
        Img_goal.vector()[:] = np.where(Img_goal.vector()[:] < 0, 0, Img_goal.vector()[:])


else:
    from mri_utils.MRI2FEM import read_image
    hyperparameters["input"]= FName
    (mesh, Img, NumData) = read_image(hyperparameters, name="input", mesh=None, printout=True, normalize=hyperparameters["normalize"])

    hyperparameters["target"]= FName_goal
    (mesh_goal, Img_goal, NumData_goal) = read_image(hyperparameters, name="target", mesh=mesh, printout=True, normalize=hyperparameters["normalize"])


# initialize trafo
vCG = VectorFunctionSpace(mesh, "CG", 1)
s1 = TrialFunction(vCG)
s2 = TestFunction(vCG)
form = inner(s1, s2) * dx
mass_action_form = action(form, Constant((1., 1.)))
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
controlf = transformation(controlfun, M_lumped_inv)
control = preconditioning(controlf)#, smoothen=True)
control.rename("control", "")

# parameters
# DeltaT = 1e-3
MaxIter = int(hyperparameters["maxiter"])
alpha = Constant(hyperparameters["alpha"]) #regularization
DeltaT = 1 / MaxIter

maxlbfgsiter = int(hyperparameters["maxlbfs"])

if hyperparameters["newTransport"]:
    print("Using new implementation with Crank Nicolson")
    from DGTransport import Transport
    
    Img_deformed = Transport(Img, control, hyperparameters=hyperparameters,
                            MaxIter=MaxIter, DeltaT=DeltaT, timestepping=hyperparameters["timestepping"], 
                            solver=hyperparameters["solver"], MassConservation=False)

else:
    print("Using original implementation")
    from DGTransportOrig import Transport
    Img_deformed = Transport(Img, control, MaxIter, DeltaT, MassConservation = False)


#File("output" + filename + "/test.pvd") << Img_deformed

# solve forward and evaluate objective


state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
cont = Control(controlfun)

if hyperparameters["regularize"] == "L2control":
    J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx + alpha*controlf**2*dx(domain=mesh))

elif hyperparameters["regularize"] == "velocity":

    J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx + alpha*control**2*dx(domain=mesh))

Jhat = ReducedFunctional(J, cont)

files = {}
files["lossfile"] = 'loss.txt'
files["regularizationfile"] = 'regularization.txt'


optimization_iterations = 0
def cb(*args, **kwargs):
    global optimization_iterations

    print(optimization_iterations, "/", maxlbfgsiter)
    optimization_iterations += 1
    current_pde_solution = state.tape_value()
    current_pde_solution.rename("Img", "")


    current_control = cont.tape_value()

    if current_pde_solution.vector()[:].max() > 10:
        print("State became > 10 at some vertex, something is probably wrong")

    Jd = assemble(0.5 * (current_pde_solution - Img_goal)**2 * dx(domain=Img.function_space().mesh()))
    Jreg = assemble(alpha*(current_control)**2*dx(domain=Img.function_space().mesh()))

    if MPI.rank(MPI.comm_world) == 0:
    
        with open(files["lossfile"], "a") as myfile:
            myfile.write(str(float(Jd))+ ", ")
        with open(files["regularizationfile"], "a") as myfile:
            myfile.write(str(float(Jreg))+ ", ")

        # print("Wrote to lossfile and regularizationfile, stored Jd_current")
    
    hyperparameters["Jd_current"] = float(Jd)
    hyperparameters["Jv_current"] = float(Jreg)

    # current_control = cont.tape_value()
    # current_control.rename("control", "")

    # current_trafo.rename("transformation", "")

    # fCont.write(current_control, float(optimization_iterations))
    # fTrafo.write(current_trafo, float(optimization_iterations))
    # fState.write(current_pde_solution, float(optimization_iterations))
    
    FName = "output" + filename + "/optimize" + format(optimization_iterations, ".0f") + ".png"
    FEM2Pic(current_pde_solution, NumData, FName)
  
# fState.write(Img_deformed, float(0))
# fCont.write(control, float(0))

minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": maxlbfgsiter}, tol=1e-08, callback = cb)

with open('hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)



confun = Function(vCG)
confun.vector().set_local(controlfun.vector())
confun = transformation(confun, M_lumped_inv)
confun = preconditioning(confun) # , smoothen=True)
File("output" + filename + "/OptControl.pvd") << confun


stateFile = HDF5File(MPI.comm_world, "State.hdf", "w")
stateFile.write(mesh, "mesh")
stateFile.write(state.tape_value(), "-1")

velocityFile = HDF5File(MPI.comm_world, "VelocityField.hdf", "w")
velocityFile.write(mesh, "mesh")
velocityFile.write(confun, "-1")
"""
h = Function(vCG)
h.vector()[:] = 0.1
h.vector().apply("")
conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)
"""