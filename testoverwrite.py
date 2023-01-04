from fenics import *
from fenics_adjoint import *
import os

set_log_level(LogLevel.CRITICAL)
# set_log_level(20)
parameters['ghost_mode'] = 'shared_facet'

domainmesh = UnitSquareMesh(MPI.comm_world, 4, 4)
V1 = FunctionSpace(domainmesh, "DG", 1)

# velocityFile = HDF5File(domainmesh.mpi_comm(), "u.hdf", "w")
velocityFile = HDF5File(MPI.comm_world, "u.hdf", "w")
velocityFile.write(domainmesh, "mesh")


#velocityFile.parameters["flush_output"] = True
#velocityFile.parameters["rewrite_function_mesh"] = False

for i in range(5):

    u = Function(V1)

    u.vector()[:] = i
    print("Creating, i=", i, assemble(u*dx))

    # u.rename("", "")
    # u = Control(u)
    # u = u.tape_value()
    # u.vector().update_ghost_values()
    
    # velocityFile.write(u, str(i))
    # velocityFile.flush()
    
    velocityFile.write(u, "test")
    
    

velocityFile.close()

del u, V1, domainmesh

domainmesh = Mesh()
hdf = HDF5File(domainmesh.mpi_comm(), "./u.hdf", "r")
hdf.read(domainmesh, "mesh", False)
V1 = FunctionSpace(domainmesh, "DG", 1)

v = Function(V1)

hdf.read(v, "test")

print("Reading", assemble(v*dx))


os.system("rm ./u.hdf")