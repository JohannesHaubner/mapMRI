from fenics import *
from fenics_adjoint import *

parameters['allow_extrapolation'] = True

coarsemesh = UnitSquareMesh(4,4)

#finemesh = refine(coarsemesh)
#print("refined mesh")
n = 2
finemesh = UnitSquareMesh(n * 4, n * 4)

Vcoarse = FunctionSpace(coarsemesh, "CG", 1)
Vfine = FunctionSpace(finemesh, "CG", 1)

u_coarse = Function(Vcoarse)

# expression = Expression("f", f=u_coarse, degree=1)

# print("Trying to interpolate")
# u_inter = interpolate(expression, Vfine)

print("Trying to project function")
# u_inter = project(expression, Vfine)
u_inter = project(u_coarse, Vfine)

print("Managed to interpolate, script finished")