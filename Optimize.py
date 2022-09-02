from dolfin import *
from dolfin_adjoint import *
from DGTransport import Transport
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
fcont = XDMFFile(MPI.comm_world, "output/Control.xdmf")
fcont.parameters["flush_output"] = True
fcont.parameters["rewrite_function_mesh"] = False

fphi = XDMFFile(MPI.comm_world, "output/Phi.xdmf")
fphi.parameters["flush_output"] = True
fphi.parameters["rewrite_function_mesh"] = False

writeiter = 0

class AdjointWriter():
    def __init__(self, phi, png=False):
        self.writeiter = 0
        self.js = []
        self.phi = phi
        self.png = png
    def eval(self, j, control):
        if self.writeiter % 10 == 0 or self.writeiter < 10:
            fcont.write_checkpoint(control, "control", float(self.writeiter), append=True)
            fphi.write_checkpoint(self.phi, "phi", float(self.writeiter), append=True)
        if self.png: 
            FEM2Pic(self.phi, 1, "output/phi"+str(self.writeiter)+".png")
        self.js += [j]
        print("objective function: ", j)
        self.writeiter += 1

# transform colored image to black-white intensity image
DG = FunctionSpace(mesh, "DG", 1)
Img = project(sqrt(inner(Img, Img)), DG)
Img.rename("img", "")
Img_goal = project(sqrt(inner(Img_goal, Img_goal)), DG)

set_working_tape(Tape())

# function spaces and definitions
vCG = VectorFunctionSpace(mesh, "CG", 1)

controlfun = Function(vCG)
x = SpatialCoordinate(mesh)
controlfun = project(as_vector((0.0, x[1])), vCG)
#controlfun.vector().set_local(Wind_data.vector())
#controlfun.vector().apply("")

control = preconditioning(controlfun)

# parameters
DeltaT = 2e-4
MaxIter = 500

#Img_deformed = Transport(Img, control, MaxIter, DeltaT, MassConservation = False)
Img_deformed = Transport(Img, controlfun, MaxIter, DeltaT, MassConservation = False)

File("output/test.pvd") << Img_deformed

# solve forward and evaluate objective
source = Function(DG)

alpha = Constant(1e-4)
J = assemble(0.5 * (Img - Img_goal)**2 * dx + alpha*grad(control)**2*dx)

mycallback = AdjointWriter(Img, png=True).eval
Jhat = ReducedFunctional(J, Control(controlfun), eval_cb_post=mycallback)

minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True}, tol=1e-08)

"""
h = Function(vCG)
h.vector()[:] = 0.1
h.vector().apply("")
conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)
"""
