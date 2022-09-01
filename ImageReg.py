from dolfin import *
from Pic2Fen import *
parameters['ghost_mode'] = 'shared_facet'
from dolfin_adjoint import *
from preconditioning_overloaded import preconditioning
import numpy
set_log_level(30)

# read image
FName = "shuttle_small.png"
(mesh, Img, NumData) = Pic2Fenics(FName)

FName_goal = "shuttle_goal.png"
(mesh_goal, Img_goal, NumData_goal) = Pic2Fenics(FName_goal, mesh)

# output file
fout = XDMFFile(MPI.comm_world, "output/Result.xdmf")
fout.parameters["flush_output"] = True
fout.parameters["rewrite_function_mesh"] = False

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

control = Function(vCG)
control.vector().set_local(Wind_data.vector())
control.vector().apply("")
preconditioning(control)

File("mycontrol.pvd") << control

J = assemble( control**2*dx)
Jhat = ReducedFunctional(J, Control(control))

h = Function(control.function_space())
h.vector()[:] = 2.0
h.vector().apply("")
#BC.apply(h.vector())

conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)
exit()

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
solver = LUSolver(A, "mumps")
for i in range(2):
    b = assemble(rhs(a))
    b.apply("")
    solver.solve(Img.vector(), b)
    Img_old.assign(Img)

J = assemble(0.5 * (Img - Img_goal)**2 * dx)
Jhat = ReducedFunctional(J, Control(control))

#from IPython import embed; embed()
h = Function(control.function_space())
h.vector()[:] = 2.0
h.vector().apply("")
#BC.apply(h.vector())

conv_rate = taylor_test(Jhat, control, h)
print(conv_rate)

#minimize(Jhat,  method = 'L-BFGS')
