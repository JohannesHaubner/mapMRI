from dolfin import *
from dolfin_adjoint import *
from DGTransport import Transport
from ipopt_solver import IPOPTProblem as IpoptProblem
from ipopt_solver import IPOPTSolver as IpoptSolver
#from SUPGTransport import Transport
from Pic2Fen import *

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

# set option
smoothen = True

set_working_tape(Tape())

#initialize control
vCG = VectorFunctionSpace(mesh, "CG", 1)
controlfun = Function(vCG)
#x = SpatialCoordinate(mesh)
#controlfun = project(as_vector((0.0, x[1])), vCG)

control = preconditioning(controlfun, smoothen=smoothen)
control.rename("control", "")

# parameters
DeltaT = 1e-3
MaxIter = 50

Img_deformed = Transport(Img, control, MaxIter, DeltaT, MassConservation = False)

#File("output" + filename + "/test.pvd") << Img_deformed

# solve forward and evaluate objective
alpha = Constant(1e-3) #regularization

state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
cont = Control(controlfun)
print(type(Img_deformed))
print(type(Img_goal))
print(type(control))
print(type(mesh))
if not smoothen:
    J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx + alpha*grad(control)**2*dx(domain=mesh))
else:
    J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx)

Jhat = ReducedFunctional(J, cont)

optimization_iterations = 0
def cb(*args, **kwargs):
    global optimization_iterations
    optimization_iterations += 1
    current_pde_solution = state.tape_value()
    current_pde_solution.rename("Img", "")
    current_control = cont.tape_value()
    current_control.rename("control", "")

    fCont.write(current_control, float(optimization_iterations))
    fState.write(current_pde_solution, float(optimization_iterations))
    
    FName = "output" + filename + "/optimize_%5d.png"%optimization_iterations
    FEM2Pic(current_pde_solution, NumData, FName)
  
fState.write(Img_deformed, float(0))
fCont.write(control, float(0))

if smoothen:
    # IPOPT
    s1 = TrialFunction(vCG)
    s2 = TestFunction(vCG)
    form = inner(s1, s2) * dx
    mass_action_form = action(form, Constant((1., 1.)))
    mass_action = assemble(mass_action_form)
    ndof = mass_action.size()
    diag_entries = mass_action.gather(range(ndof))
    problem = IpoptProblem([Jhat], [1.0], [], [], [], [], diag_entries, alpha.values()[0])
    ipopt = IpoptSolver(problem, callback = cb)
    controlfun = ipopt.solve(cont.vector()[:])
else:
    minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": maxiter}, tol=1e-08, callback = cb)

confun = Function(vCG)
confun.vector().set_local(controlfun)
File("output" + filename + "/OptControl.pvd") << confun

"""
h = Function(vCG)
h.vector()[:] = 0.1
h.vector().apply("")
conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)
"""
