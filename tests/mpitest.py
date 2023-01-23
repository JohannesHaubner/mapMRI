from fenics import *
import time
m = UnitSquareMesh(8, 8)
V = FunctionSpace(m, "CG", 1)

u = Function(V)
#  v = interpolate(Expression("x[0]", degree=1), V)
v = Function(V)

if MPI.rank(MPI.comm_world) == 0:
    # v.vector()[:] = 1
    pass

# time.sleep(1)

print(MPI.rank(MPI.comm_world), u.vector()[:]-v.vector()[:])