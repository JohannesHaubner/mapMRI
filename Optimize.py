from dolfin import *
from dolfin_adjoint import *
#from DGTransport import Transport
from SUPGTransport import Transport
from Pic2Fen import *

from preconditioning_overloaded import preconditioning

import numpy
set_log_level(20)

# read image
FName = "shuttle_small.png"
(mesh, Img, NumData) = Pic2FEM(FName)

FName_goal = "shuttle_goal.png"
(mesh_goal, Img_goal, NumData_goal) = Pic2FEM(FName_goal, mesh)

# output file
fCont = XDMFFile(MPI.comm_world, "output/Control.xdmf")
fCont.parameters["flush_output"] = True
fCont.parameters["rewrite_function_mesh"] = False

fState = XDMFFile(MPI.comm_world, "output/State.xdmf")
fState.parameters["flush_output"] = True
fState.parameters["rewrite_function_mesh"] = False

"""
writeiter = 0
class AdjointWriter():
    def __init__(self, phi, png=False):
        self.writeiter = 0
        self.js = []
        self.phi = phi
        self.png = png
    def eval(self, j, control):
        if self.writeiter % 10 == 0 or self.writeiter < 10:
            #fCont.write_checkpoint(control, "control", float(self.writeiter), append=True)
            control.rename("control", "")
            fCont.write(control, float(self.writeiter))
            fState.write_checkpoint(self.phi, "phi", float(self.writeiter), append=True)
        if self.png: 
            FEM2Pic(self.phi, 1, "output/phi"+str(self.writeiter)+".png")
        self.js += [j]
        print("objective function: ", j)
        self.writeiter += 1
"""

# transform colored image to black-white intensity image
#Space = FunctionSpace(mesh, "DG", 1)
Space = FunctionSpace(mesh, "CG", 1)
Img = project(sqrt(inner(Img, Img)), Space)
Img.rename("img", "")
Img_goal = project(sqrt(inner(Img_goal, Img_goal)), Space)

set_working_tape(Tape())

# function spaces and definitions
vCG = VectorFunctionSpace(mesh, "CG", 1)

#initialize control
controlfun = Function(vCG)
#x = SpatialCoordinate(mesh)
#controlfun = project(as_vector((0.0, x[1])), vCG)

control = preconditioning(controlfun)

# parameters
DeltaT = 4e-4
MaxIter = 200

Img_deformed = Transport(Img, control, MaxIter, DeltaT, MassConservation = False)

#File("output/test.pvd") << Img_deformed

# solve forward and evaluate objective
alpha = Constant(1e-3) #regularization

state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
cont = Control(controlfun)
J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx + alpha*grad(control)**2*dx(domain=mesh))

Jhat = ReducedFunctional(J, cont)

optimization_iterations = 0
def cb(*args, **kwargs):
    global optimization_iterations
    optimization_iterations += 1
    current_pde_solution = state.tape_value()
    current_pde_solution.rename("Img", "")
    current_control = cont.tape_value()
    current_control.rename("control", "")
    
    #File("output/control_iter{}.pvd".format(optimization_iterations)) << current_contro
    fCont.write(current_control, float(optimization_iterations))
    #File("output/state_iter{}.pvd".format(optimization_iterations)) << current_pde_solution
    fState.write(current_pde_solution, float(optimization_iterations))
  
fState.write(Img_deformed, float(0))
fCont.write(control, float(0))
    
minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True}, tol=1e-08, callback = cb)

File("output/OptControl.pvd") << controlfun

"""
h = Function(vCG)
h.vector()[:] = 0.1
h.vector().apply("")
conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)
"""
