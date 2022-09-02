from dolfin import *
from Pic2Fen import *
parameters['ghost_mode'] = 'shared_facet'
from dolfin_adjoint import *
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

class myclass():
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
            FEM2Pic(self.phi, 1, "/output/pics/phi"+str(self.writeiter)+".png")
        self.js += [j]
        print("objective function: ", j)
        self.writeiter += 1
        

# function spaces and definitions
DG = FunctionSpace(mesh, "DG", 1)
vCG = VectorFunctionSpace(mesh, "CG", 1)
n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
v = TestFunction(DG)

# transform colored image to black-white intensity image
Img = project(sqrt(inner(Img, Img)), DG)
Img.rename("img", "")
Img_goal = project(sqrt(inner(Img_goal, Img_goal)), DG)

# define initial velocity field
BC = DirichletBC(vCG, Constant((0.0,0.0)), "on_boundary")
Wind_data = Function(vCG)
mylhs = inner(grad(TestFunction(vCG)), grad(TrialFunction(vCG))) * dx
myrhs = inner(as_vector([sin(x[0]/100), cos(x[1]/100)]), TestFunction(vCG)) * dx
solve(mylhs == myrhs, Wind_data, BC)

set_working_tape(Tape())

controlfun = Function(vCG)
#controlfun.vector().set_local(Wind_data.vector())
#controlfun.vector().apply("")


control = preconditioning(controlfun)

#BC.apply(control.vector())

#Make form:
#from IPython import embed; embed()

def Max0(d):
    """
    val = []
    for i in range(NumData):
        val.append(0.5*(d[i]+abs(d[i])))
    return as_vector(val)
    """

    return 0.5*(d+abs(d))

def Flux(f, Wind, n):
    upwind = Max0(inner(Wind,n))
    return -f*upwind


def Form(f, v, Wind):
    a = inner(grad(v), outer(f, Wind)) * dx
    a += inner(jump(v), jump(Flux(f, Wind, n))) * dS
    a += inner(v, Flux(f, Wind, n)) * ds
    a += div(Wind) * inner(v, f) * dx
    return a

# parameters
DeltaT = 1e-5

# solve forward and evaluate objective
source = Function(DG)

Img_t = TrialFunction(Img.function_space())
Img_old = Function(Img.function_space())
Img_old.assign(Img)

a = Constant(1.0/DeltaT)*(inner(v, Img_t) * dx - inner(v, Img_old) * dx) \
    - 0.5*(Form(Img_old, v, control) + Form(Img_t, v,  control))

A = assemble(lhs(a))
#solver = LUSolver(A, "mumps")
solver = KrylovSolver("gmres", "none")
solver.set_operator(A)
for i in range(600):
    b = assemble(rhs(a))
    b.apply("")
    solver.solve(Img.vector(), b)
    Img_old.assign(Img)



alpha = Constant(1e-4)
J = assemble(0.5 * (Img - Img_goal)**2 * dx + alpha*grad(control)**2*dx)


mycallback = myclass(Img).eval
Jhat = ReducedFunctional(J, Control(controlfun), eval_cb_post=mycallback)

minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True}, tol=1e-08)

