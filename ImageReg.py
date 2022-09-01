from dolfin import *
from Pic2Fen import *
parameters['ghost_mode'] = 'shared_facet'
from dolfin_adjoint import *
import numpy
set_log_level(30)

# read image
FName = "shuttle_small.png"
(mesh, Img, NumData) = Pic2Fenics(FName)

FName_goal = "shuttle_goal.png"
(mesh_goal, Img_goal, NumData_goal) = Pic2Fenics(FName_goal)

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

def Form(f, v, Wind, source=0):
    a = inner(grad(v), outer(f, Wind)) * dx
    a += inner(jump(v), jump(Flux(f, Wind, n))) * dS
    a += inner(v, Flux(f, Wind, n)) * ds
    a += div(Wind) * inner(v, f) * dx
    a += source * v * dx(domain=mesh)
    return a

# parameters
DeltaT = 1e-5

# solve forward and evaluate objective
source = Function(DG)

Img_old = Function(Img.function_space())
Img_old.assign(Img)

a = Constant(1.0/DeltaT)*(inner(v, Img) * dx - inner(v, Img_old) * dx) \
    - 0.5*(Form(Img_old, v, Wind_data, source) + Form(Img, v,  Wind_data, source))

A = assemble(lhs(a))
solver = LUSolver(A, "mumps")
for i in range(2):
    b = assemble(rhs(a))
    b.apply("")
    solver.solve(Img.vector(), b)
    Img_old.assign(Img)

J = assemble(0.5 * (Img - Img_goal)**2 * dx)
Jhat = ReducedFunctional(J, Control(source))

#from IPython import embed; embed()
h = Function(DG)
h.vector()[:] = 0.1
conv_rate = taylor_test(Jhat, source, h)
print(conv_rate)