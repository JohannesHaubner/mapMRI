from dolfin import *
from DGTransport import solve as DGSolve

from Pic2Fen import *

import numpy
set_log_level(30)
import time

FName = "shuttle_small.png"
(mesh, Img, NumData) = Pic2Fenics(FName)

fout = XDMFFile(MPI.comm_world, "output/Result.xdmf")
fout.parameters["flush_output"] = True
fout.parameters["rewrite_function_mesh"] = False

# some defs
DG = FunctionSpace(mesh, "DG", 1)
vCG = VectorFunctionSpace(mesh, "CG", 1)
n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
v = TestFunction(DG)


Img = project(sqrt(inner(Img, Img)), DG)
Img.rename("img", "")


Img_deformed = project(sqrt(inner(Img, Img)), DG)

#mark boundary edges as 1
BoundaryMarker = MeshFunction("size_t", mesh, 1)
for e in edges(mesh):
    if len(e.entities(2)) == 1:
        BoundaryMarker[e.index()] = 1

#make v for defomed data
BC = DirichletBC(vCG, Constant((0.0,0.0)), BoundaryMarker, 1)
Wind_data = Function(vCG)
mylhs = inner(grad(TestFunction(vCG)), grad(TrialFunction(vCG)))*dx
myrhs = inner(as_vector([sin(x[0]/100), cos(x[1]/100)]), TestFunction(vCG))*dx
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
    a = inner(grad(v), outer(f, Wind))*dx
    a += inner(jump(v), jump(Flux(f, Wind, n)))*dS
    a += inner(v, Flux(f, Wind, n))*ds
    a += div(Wind)*inner(v, f)*dx
    a += source*v*dx(domain=mesh)
    return a

Img_next = TrialFunction(Img.function_space())
#Img_next = Function(Img.function_space())
#Img_next.rename("img", "")
DeltaT = 1e-5
a_deformed = Constant(1.0/DeltaT)*(inner(v,Img_next)*dx - inner(v, Img_deformed)*dx) - 0.5*(Form(Img_deformed, v, Wind_data) + Form(Img_next, v,  Wind_data))
#a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(f_next)
#a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(Img)



A = assemble(lhs(a_deformed))
solver = LUSolver(A, "mumps")
#from IPython import embed; embed()

for i in range(500):
    fout.write(Img_deformed, float(i))
    b = assemble(rhs(a_deformed))
    b.apply("")
    solver.solve(Img.vector(), b)

# opt
print("finished creating tracking")
Wind_opt = Function(vCG)
source = Function(DG)
source.vector().set_local(numpy.random.rand(source.vector().get_local().size))
source.vector().apply("")


DeltaT = 1e-5
a = Constant(1.0/DeltaT)*(inner(v,Img_next)*dx - inner(v, Img)*dx) - 0.5*(Form(Img, v, Wind_opt, source) + Form(Img_next, v,  Wind_opt, source))

A = assemble(lhs(a))
solver = LUSolver(A, "mumps")
for i in range(20):
    b = assemble(rhs(a_deformed))
    b.apply("")
    solver.solve(Img.vector(), b)
    #solve( lhs(a) == rhs(a), Img)
    
# get v = 0 on boundary bound vec
dofmap = vCG.dofmap()
boundvec = numpy.ones(Wind_opt.vector().get_local().size)*1e+10
for e in edges(mesh):
    if len(e.entities(2) == 1):
        for v in e.entities(0):
            for dof in dofmap.entity_dofs(mesh, 0, [v]):
                if dof < boundvec.size:
                    boundvec[dof] = 0


Boundlowfun = Function(vCG)
Boundlowfun.vector().set_local(-boundvec)
Boundlowfun.vector().apply("")
Boundhighfun = Function(vCG)
Boundhighfun.vector().set_local(boundvec)
Boundhighfun.vector().apply("")


J = assemble( 0.5*(Img-Img_deformed)**2*dx)
Jhat = ReducedFunctional(J, Control(source))

#from IPython import embed; embed()
h = Function(DG)
h.vector()[:] = 0.1
conv_rate = taylor_test(Jhat, source, h)
print(conv_rate)

#m_opt = minimize(Jhat, method = 'L-BFGS-B', bounds=(Boundlowfun, Boundhighfun))




