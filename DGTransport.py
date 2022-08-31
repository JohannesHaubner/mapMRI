#from dolfin import *
from dolfin import *
parameters['ghost_mode'] = 'shared_facet'

#info(parameters, True)
#exit()
#parameters["ghost_mode"] = "shared_vertex"

#from Pic2Fen import *

FName = "shuttle_small.png"

#create on the fly
#(mesh, Img, NumData) = Pic2Fenics(FName)

#read from file
FIn = HDF5File(MPI.comm_world, FName+".h5", 'r')
mesh = Mesh()
FIn.read(mesh, "mesh", False)
Space = VectorFunctionSpace(mesh, "DG", 0)
Img = Function(Space)
FIn.read(Img, "Data1")

"""
#make artificial
mesh = UnitSquareMesh(100,100)
x = SpatialCoordinate(mesh)
Img = project(x[0], FunctionSpace(mesh, "DG", 0))
NumData = 1
"""

Img = project(sqrt(inner(Img, Img)), FunctionSpace(mesh, "DG", 0))

"""
#n = FacetNormal(mesh)
f = project(Constant((1.0,1.0,1.0)), VectorFunctionSpace(mesh, "DG", 1, 3))
print(assemble(inner(f("+")-f("-"),f("+")-f("-"))*dS(domain=mesh)))
exit()
"""

#Img = project(Img, VectorFunctionSpace(mesh, "CG", 1, NumData))
#Img = project(Img, VectorFunctionSpace(mesh, "DG", 2, NumData))

Space = Img.function_space()

v = TestFunction(Space)

FOut = File("output/test.pvd")

Img.rename("img", "")
FOut << Img

x = SpatialCoordinate(mesh)
Wind = as_vector((0.0, x[1]))
#Wind = Constant((250.0, 250.0))
Wind = project(Wind, VectorFunctionSpace(mesh, "CG", 1))

#Make form:
n = FacetNormal(mesh)

#from IPython import embed; embed()

def Max0(d):
    """
    val = []
    for i in range(NumData):
        val.append(0.5*(d[i]+abs(d[i])))
    return as_vector(val)
    """
    #return d
    return 0.5*(d+abs(d))

def Flux(f, Wind, n):
    upwind = Max0(inner(Wind,n))
    return -f*upwind
    #return outer(f, -upwind)
    #return 0.5*dot(outer(f, -Wind), n)

def Form(f):
    #a = inner(v, div(outer(f, -Wind)))*dx
    
    a = inner(grad(v), outer(f, Wind))*dx
    a += inner(jump(v), jump(Flux(f, Wind, n)))*dS
    
    a += inner(v, Flux(f, Wind, n))*ds
    
    return a
"""
import numpy
HDT = FunctionSpace(mesh, "HDiv Trace", 0)
testfun = Function(HDT)
testfun.vector().set_local(numpy.random.rand(testfun.vector().get_local().size))
testfun.vector().apply("")

testhdt = TestFunction(HDT)
trialhdt = TrialFunction(HDT)

newfun = Function(HDT)

DG = FunctionSpace(mesh, "DG", 0)
testfunDG = Function(DG)
testfunDG.vector().set_local(numpy.random.rand(testfunDG.vector().get_local().size))
testfunDG.vector().apply("")


solve( inner(testhdt('+'), trialhdt('+'))*dS + inner(testhdt, trialhdt)*ds == inner(testfunDG('+'), testhdt('+'))*dS, newfun)
print(assemble( (newfun('+')-testfunDG('+'))**2*dS))
exit()
"""
"""
fluxp = Function(HDT)
fluxm = Function(HDT)


print( assemble( (Flux(1, Wind, n)('-') - fluxp('+'))**2*dS))
exit()
"""
Img_next = TrialFunction(Img.function_space())
#Img_next = Function(Img.function_space())
#Img_next.rename("img", "")
DeltaT = 5e-4
a = Constant(1.0/DeltaT)*(inner(v,Img_next)*dx - inner(v, Img)*dx) - 0.5*(Form(Img) + Form(Img_next))

#a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(f_next)
#a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(Img)




A = assemble(lhs(a))
#solver = LUSolver()
solver = KrylovSolver("gmres", "none")
solver.set_operator(A)
#from IPython import embed; embed()
for i in range(100):
    print(i)
        #solve(a==0, Img_next)


    b = assemble(rhs(a))
    b.apply("")
    solver.solve(Img.vector(), b)
    #Img.assign(Img_next)
    #Img.vector().set_local(Img_next.vector().get_local())
    #Img.vector().apply("")
    #Img = Img_next
    #Img = project(Img_next, Space)
    #Img_plot = project(Img, FunctionSpace(mesh, "CG", 1))
    #Img_plot.rename("img","")
    FOut << Img
    #FOut << Img_next
    #exit()
