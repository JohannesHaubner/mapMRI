from dolfin import *
from dolfin_adjoint import *

import config
hyperparameters = {}
hyperparameters["smoothen"] = True
hyperparameters["solver"] = "krylov"
hyperparameters["preconditioner"] = "amg"
config.hyperparameters = hyperparameters


from ipopt_solver import IPOPTProblem as IpoptProblem
from ipopt_solver import IPOPTSolver as IpoptSolver
#from SUPGTransport import Transport
from Pic2Fen import *

from transformation_overloaded import transformation
from preconditioning_overloaded import preconditioning

import numpy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cube", action="store_true", default=False)
parser.add_argument("--newTransport", action="store_true", default=False)
parserargs= vars(parser.parse_args())


set_log_level(20)

# read image
# filename = "mask_only"
filename = ""
if parserargs["cube"]:
    print("Using Cube as test images")
    FName = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d/input.mgz"
    FName_goal = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d/target.mgz"
else:
    print("Using rocket")
    FName = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/shuttle_small.png"
    FName_goal = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/shuttle_goal.png"



(mesh, Img, NumData) = Pic2FEM(FName)


#FName_goal = "./braindata_2d/slice091" + filename +".png"
(mesh_goal, Img_goal, NumData_goal) = Pic2FEM(FName_goal, mesh)

# # output file
# fTrafo = XDMFFile(MPI.comm_world, "output" + filename + "/Trafo.xdmf")
# fTrafo.parameters["flush_output"] = True
# fTrafo.parameters["rewrite_function_mesh"] = False

# fCont = XDMFFile(MPI.comm_world, "output" + filename + "/Control.xdmf")
# fCont.parameters["flush_output"] = True
# fCont.parameters["rewrite_function_mesh"] = False

# fState = XDMFFile(MPI.comm_world, "output" + filename + "/State.xdmf")
# fState.parameters["flush_output"] = True
# fState.parameters["rewrite_function_mesh"] = False

"""
Space = VectorFunctionSpace(mesh, "DG", 1, 3)
Img = project(Img, Space)
"""

# transform colored image to black-white intensity image
Space = FunctionSpace(mesh, "DG", 1)
Img = project(sqrt(inner(Img, Img)), Space)
Img.rename("img", "")
Img_goal = project(sqrt(inner(Img_goal, Img_goal)), Space)
NumData = 1


for u_data in [Img, Img_goal]:
    u_data.vector()[:] *= 1 / u_data.vector()[:].max()

    u_data.vector()[:] = np.where(u_data.vector()[:] < 0, 0, u_data.vector()[:])

assert np.max(Img) <= 1
assert np.min(Img_goal) >= 0

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
MaxIter = 800
alpha = Constant(1e-4) #regularization
DeltaT = 1 / MaxIter

maxlbfgsiter = 1000

if parserargs["newTransport"]:
    print("Using new implementation with Crank Nicolson")
    from DGTransport import Transport
    hyperparameters["timestepping"] = "CrankNicolson"
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
        raise ValueError("State became > 10 at some vertex, something is probably wrong")

    Jd = assemble(0.5 * (current_pde_solution - Img_goal)**2 * dx(domain=Img.function_space().mesh()))
    Jreg = assemble(alpha*(current_control)**2*dx(domain=Img.function_space().mesh()))

    if MPI.rank(MPI.comm_world) == 0:
    
        with open(files["lossfile"], "a") as myfile:
            myfile.write(str(float(Jd))+ ", ")
        with open(files["regularizationfile"], "a") as myfile:
            myfile.write(str(float(Jreg))+ ", ")

        # print("Wrote to lossfile and regularizationfile, stored Jd_current")
    
    hyperparameters["Jd_current"] = float(Jd)
    hyperparameters["Jreg_current"] = float(Jreg)

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