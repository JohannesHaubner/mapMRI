from fenics import *
from fenics_adjoint import *
parameters['ghost_mode'] = 'shared_facet'

n = 32
coarsemesh = UnitCubeMesh(n, n, n)

# finemesh = refine(coarsemesh) # , redistribute=False)

Vcoarse = VectorFunctionSpace(coarsemesh, "CG", 1)
# Vfine = FunctionSpace(finemesh, "DG", 1)

u_coarse = interpolate(Expression(("x[0]*x[1]", "0", "0"), degree=2), Vcoarse)

# u_coarse.set_allow_extrapolation(True)
# u_coarse.rename("u_coarse", "")
# u_coarse.vector().update_ghost_values()

# print("Trying to interpolate")
# u_inter = Function(Vfine)
# u_inter.interpolate(u_coarse)
# u_inter.vector().update_ghost_values()

# print("Managed to interpolate, storing files")


with XDMFFile("coarse.xdmf") as xdmf:
    xdmf.write_checkpoint(u_coarse, "coarse", 0.)

# with XDMFFile("fine.xdmf") as xdmf:
#     xdmf.write_checkpoint(u_inter, "fine", 0.)


controlFile = HDF5File(coarsemesh.mpi_comm(), "Coarse.hdf", "w")
controlFile.write(coarsemesh, "mesh")
controlFile.write(u_coarse, "coarse")
controlFile.close()