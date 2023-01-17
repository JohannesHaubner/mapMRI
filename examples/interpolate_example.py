from fenics import *

Lx = 4
Ly = 3

coarsemesh = RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(Lx, Ly), Lx, Ly)

nx = 2 * Lx
ny = 3 * Ly
finemesh = RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(Lx, Ly), nx, ny)

Vcoarse = FunctionSpace(coarsemesh, "CG", 1)
Vfine = FunctionSpace(finemesh, "CG", 1)

u_coarse = Function(Vcoarse)

print("Trying to project function")

u_inter = project(u_coarse, Vfine)

print("Managed to interpolate, script finished")