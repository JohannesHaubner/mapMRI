from fenics import *
from fenics_adjoint import *

coarsemesh = UnitSquareMesh(4,4)

finemesh = refine(coarsemesh)
print("refined mesh")
# n = 2
# finemesh = UnitSquareMesh(n * 4, n * 4)

Vcoarse = FunctionSpace(coarsemesh, "DG", 1)
Vfine = FunctionSpace(finemesh, "DG", 1)

u_coarse = Function(Vcoarse)

expression = Expression("f", f=u_coarse, degree=1)

print("Trying to interpolate")
u_inter = interpolate(expression, Vfine)

print("Managed to interpolate, script finished")