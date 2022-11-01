from dolfin import *
from dolfin_adjoint import *
from DGTransport import Transport
#from SUPGTransport import Transport
# from Pic2Fen import *
import os
from mri_utils.MRI2FEM import read_image

from preconditioning_overloaded import preconditioning

import numpy
set_log_level(20)

if "home/bastian/" in os.getcwd():
    path = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/"


# FName = path + "mridata_3d/091_cropped.mgz"
# FName_goal = path + "mridata_3d/091registeredto205.mgz"
# outputfolder = path + "outputs/output_cropped_mri"

FName = path + "testdata_3d/input.mgz"
FName_goal = path + "testdata_3d/target.mgz"
outputfolder = path + "outputs/output_cube"

if not os.path.isdir(outputfolder):
    os.makedirs(outputfolder, exist_ok=True)

maxiter = 200

(mesh, Img, NumData) = read_image(FName)


(mesh_goal, Img_goal, NumData_goal) = read_image(FName_goal, mesh)

#  breakpoint()

# output file
fCont = XDMFFile(MPI.comm_world, outputfolder + "/Control.xdmf")
fCont.parameters["flush_output"] = True
fCont.parameters["rewrite_function_mesh"] = False

fState = XDMFFile(MPI.comm_world, outputfolder + "/State.xdmf")
fState.parameters["flush_output"] = True
fState.parameters["rewrite_function_mesh"] = False

"""
Space = VectorFunctionSpace(mesh, "DG", 1, 3)
Img = project(Img, Space)
"""

# transform colored image to black-white intensity image
Space = FunctionSpace(mesh, "DG", 1)
Img = project(Img, Space)
Img.rename("img", "")
Img_goal = project(Img_goal, Space)
NumData = 1

File(outputfolder + "/input.pvd") << Img
File(outputfolder + "/target.pvd") << Img_goal

set_working_tape(Tape())

#initialize control
vCG = VectorFunctionSpace(mesh, "CG", 1)
controlfun = Function(vCG)
#x = SpatialCoordinate(mesh)
#controlfun = project(as_vector((0.0, x[1])), vCG)

control = preconditioning(controlfun)
control.rename("control", "")

# parameters
DeltaT = 1e-3
MaxIter = 50

Img_deformed = Transport(Img, control, MaxIter, DeltaT, MassConservation = False)

#File( outputfolder + "/test.pvd") << Img_deformed

# solve forward and evaluate objective
alpha = Constant(1e-3) #regularization

state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
cont = Control(controlfun)
print(type(Img_deformed))
print(type(Img_goal))
print(type(control))
print(type(mesh))

breakpoint()
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
    
minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": maxiter}, tol=1e-08, callback = cb)

# File( outputfolder + "/OptControl.pvd") << controlfun

"""
h = Function(vCG)
h.vector()[:] = 0.1
h.vector().apply("")
conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)
"""
