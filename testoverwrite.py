from fenics import *
from fenics_adjoint import *
import os
# from IPython import embed
if True:
# if True:
    import h5py
    print("Imported h5py")
    os.system("echo 'Imported h5py'")


# set_log_level(LogLevel.CRITICAL)
# set_log_level(20)
parameters['ghost_mode'] = 'shared_facet'

print("log level:", get_log_level())

domainmesh = UnitSquareMesh(MPI.comm_world, 4, 4)

space= "DG"
V1 = FunctionSpace(domainmesh, space, 1)

# velocityFile = HDF5File(domainmesh.mpi_comm(), "u.hdf", "w")
velocityFile = HDF5File(MPI.comm_world, "u.hdf", "w")
velocityFile.write(domainmesh, "mesh")

for key, item in velocityFile.parameters.items():
    print(key, item)

# embed()

files = {}
files["velocityFile"] = velocityFile

#velocityFile.parameters["flush_output"] = True
#velocityFile.parameters["rewrite_function_mesh"] = False
u = Function(V1)
files["velocityFile"].write(u, str(0))
# files["velocityFile"].write(u, str(0))
# exit()

for i in range(2):

    u = Function(V1)

    u.vector()[:] = i
    print("Creating, i=", i, assemble(u*dx))


    files["velocityFile"].write(u, str(i))
    files["velocityFile"].write(u, str(99))

    with XDMFFile("Ucheck.xdmf") as xdmf:
        xdmf.write_checkpoint(u, "u", 0)
    
    

files["velocityFile"].close()

del u, V1, domainmesh

domainmesh = Mesh()
hdf = HDF5File(domainmesh.mpi_comm(), "./u.hdf", "r")
hdf.read(domainmesh, "mesh", False)
V1 = FunctionSpace(domainmesh, space, 1)

for i in range(2):
    v = Function(V1)
    vc = Function(V1)

    with XDMFFile("Ucheck.xdmf") as xdmf:
        xdmf.read_checkpoint(vc, "u", 0)

    hdf.read(v, str(i))

    print("Reading", i,  assemble(v*dx))
    print("Reading checkpoint", i,  assemble(vc*dx))

i = 99
v = Function(V1)
hdf.read(v, str(i))
print("Reading", i,  assemble(v*dx))

os.system("rm -v ./u.hdf")