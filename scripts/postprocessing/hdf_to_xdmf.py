import h5py
from fenics import *
from fenics_adjoint import *
import os

nx, ny, nz = 16, 32, 4
x0, y0, z0 = 0, 0, 0
x1, y1, z1 = 1, 2, 3    
# BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
m = BoxMesh(Point(x0, y0, z0), Point(x1, y1, z1), nx, ny, nz)
print(m.coordinates(), m.coordinates().shape)
print("exiting")
exit()

for folder in [x for x in os.listdir("./") if os.path.isdir(x)]:
    print(folder)



    f = h5py.File(folder + "/Imgh5.hdf")

    print(list(f.items()))


    mesh = Mesh()

    hdf = HDF5File(mesh.mpi_comm(), folder + "/Imgh5.hdf", "r")
    hdf.read(mesh, "mesh", False)
    V = FunctionSpace(mesh, "DG", 1)
    u = Function(V)
    hdf.read(u, "Img")
    hdf.close()

    File(folder + "/Reloaded.pvd") << u

    file = XDMFFile(MPI.comm_world, folder + "/Reloaded.xdmf")
    file.parameters["flush_output"] = True
    file.parameters["rewrite_function_mesh"] = False
    # fCont.write(Img.function_space().mesh(), '/mesh')
    file.write(u, 0)
    file.close()

    with XDMFFile(folder + "/ReloadedCheckpoint.xdmf") as xdmf:
        xdmf.write_checkpoint(u, "checkpoints", 0.)

    print("Read")