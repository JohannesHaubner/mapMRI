from fenics import *
from fenics_adjoint import *

coarsemesh = Mesh()

controlFile = HDF5File(coarsemesh.mpi_comm(), "Coarse.hdf", "r")
controlFile.read(coarsemesh, "mesh", False)

Vcoarse = VectorFunctionSpace(coarsemesh, "CG", 1)
u_coarse = Function(Vcoarse)

controlFile.read(u_coarse, "coarse")
controlFile.close()



print("Read mesh, refining")
finemesh = refine(coarsemesh, redistribute=False)
Vfine = VectorFunctionSpace(finemesh, "CG", 1)


# u_coarse.set_allow_extrapolation(True)
u_coarse.rename("u_coarse", "")
u_coarse.vector().update_ghost_values()

print("Trying to interpolate")
u_inter = interpolate(u_coarse, Vfine)
# u_inter = Function(Vfine)
# u_inter.interpolate(u_coarse)
# u_inter.vector().update_ghost_values()

print("Managed to interpolate, storing files")

with XDMFFile("coarse_reloaded.xdmf") as xdmf:
    xdmf.write_checkpoint(u_coarse, "coarse", 0.)

with XDMFFile("fine_reloaded.xdmf") as xdmf:
    xdmf.write_checkpoint(u_inter, "fine", 0.)

