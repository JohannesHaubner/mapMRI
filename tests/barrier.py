from fenics import *
import time


if MPI.rank(MPI.comm_world) == 0:
    print("mpi rank 0, sleeping")
    m = UnitCubeMesh(2,2,2)

else:
    print("rank", MPI.rank(MPI.comm_world))

print("rank", MPI.rank(MPI.comm_world), "reached barrier")
MPI.barrier(MPI.comm_world)


print("rank", MPI.rank(MPI.comm_world), "after barrier")