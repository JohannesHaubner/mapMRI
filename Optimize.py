from dolfin import *
from dolfin_adjoint import *
from DGTransport import Transport
from ipopt_solver import IPOPTProblem as IpoptProblem
from ipopt_solver import IPOPTSolver as IpoptSolver
#from SUPGTransport import Transport
from Pic2Fen import *

from transformation_overloaded import transformation
from preconditioning_overloaded import preconditioning

import numpy
set_log_level(20)

# read image
filename = "mask_only"
FName = "shuttle_small.png"
#FName = "./braindata_2d/slice205" + filename +".png"

maxiter = 1000

(mesh, Img, NumData) = Pic2FEM(FName)

FName_goal = "shuttle_goal.png"
#FName_goal = "./braindata_2d/slice091" + filename +".png"
(mesh_goal, Img_goal, NumData_goal) = Pic2FEM(FName_goal, mesh)

# output file
fTrafo = XDMFFile(MPI.comm_world, "output" + filename + "/Trafo.xdmf")
fTrafo.parameters["flush_output"] = True
fTrafo.parameters["rewrite_function_mesh"] = False

fCont = XDMFFile(MPI.comm_world, "output" + filename + "/Control.xdmf")
fCont.parameters["flush_output"] = True
fCont.parameters["rewrite_function_mesh"] = False

fState = XDMFFile(MPI.comm_world, "output" + filename + "/State.xdmf")
fState.parameters["flush_output"] = True
fState.parameters["rewrite_function_mesh"] = False

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
controlf = transformation(controlfun, M_lumped)
control = preconditioning(controlf, smoothen=True)
control.rename("control", "")

# parameters
DeltaT = 1e-3
MaxIter = 5

Img_deformed = Transport(Img, control, MaxIter, DeltaT, MassConservation = False)

#File("output" + filename + "/test.pvd") << Img_deformed

# solve forward and evaluate objective
alpha = Constant(1e-6) #regularization

state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
cont = Control(controlfun)
print(type(Img_deformed))
print(type(Img_goal))
print(type(control))
print(type(mesh))

J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx + alpha*control**2*dx(domain=mesh))

Jhat = ReducedFunctional(J, cont)

optimization_iterations = 0
def cb(*args, **kwargs):
    global optimization_iterations
    optimization_iterations += 1
    current_pde_solution = state.tape_value()
    current_pde_solution.rename("Img", "")
    current_control = cont.tape_value()
    current_control.rename("control", "")
    current_trafo = transformation(current_control, M_lumped)
    current_trafo = preconditioning(current_trafo, smoothen=True)
    current_trafo.rename("transformation", "")

    fCont.write(current_control, float(optimization_iterations))
    fTrafo.write(current_trafo, float(optimization_iterations))
    fState.write(current_pde_solution, float(optimization_iterations))
    
    FName = "output" + filename + "/optimize_%5d.png"%optimization_iterations
    FEM2Pic(current_pde_solution, NumData, FName)
  
fState.write(Img_deformed, float(0))
fCont.write(control, float(0))

h = Function(vCG)
h.vector()[:] = 0.1
h.vector().apply("")
conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)

exit()

minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": maxiter}, tol=1e-08, callback = cb)

confun = Function(vCG)
confun.vector().set_local(controlfun)
confun = transformation(confun, M_lumped)
confun = preconditioning(confun, smoothen=True)
File("output" + filename + "/OptControl.pvd") << confun

"""
h = Function(vCG)
h.vector()[:] = 0.1
h.vector().apply("")
conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)
"""
