from fenics import *
from fenics_adjoint import *
import os
import h5py
import numpy as np


def get_lumped_mass_matrix(vCG):

    s1 = TrialFunction(vCG)
    s2 = TestFunction(vCG)
    form = inner(s1, s2) * dx
    mass_action_form = action(form, Constant((1., 1., 1.)))
    M_lumped = assemble(form)
    M_lumped_inv = assemble(form)
    M_lumped.zero()
    M_lumped_inv.zero()
    diag = assemble(mass_action_form)
    diag[:] = np.sqrt(diag[:])
    diaginv = assemble(mass_action_form)
    diaginv[:] = 1.0/np.sqrt(diag[:])
    M_lumped.set_diagonal(diag)
    M_lumped_inv.set_diagonal(diaginv)

    return M_lumped

def load_velocity(hyperparameters, controlfun):

    assert os.path.isfile(hyperparameters["starting_guess"])

    print("Will try to read starting guess")
    h5file = h5py.File(hyperparameters["starting_guess"])

    print("keys in h5file", list(h5file.keys()))

    if controlfun is not None:
        print("max before loading", controlfun.vector()[:].max())
        working_mesh = controlfun.function_space().mesh()
        print("trying to read velocity without loading mesh")
        hdf = HDF5File(working_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')

        
    else:
        print("Reading mesh")
        working_mesh = Mesh()
        hdf = HDF5File(working_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
        hdf.read(working_mesh, "/mesh", False)
        
        vCG = VectorFunctionSpace(working_mesh, hyperparameters["functionspace"], hyperparameters["functiondegree"])
        controlfun = Function(vCG)

    print("trying to read", hyperparameters["readname"])
    hdf.read(controlfun, hyperparameters["readname"])
    hdf.close()

    print("max after loading", controlfun.vector()[:].max())
    print("Succesfully read starting guess")

    return working_mesh, vCG, controlfun





def interpolate_velocity(hyperparameters, domainmesh, vCG, controlfun):

    controlFile = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Loaded_Control.hdf", "w")
    controlFile.write(domainmesh, "mesh")
    controlFile.write(controlfun, "loaded_control")
    controlFile.close()

    with XDMFFile(hyperparameters["outputfolder"] + "/Loaded_Control.xdmf") as xdmf:
        xdmf.write_checkpoint(controlfun, "coarse", 0.)

    domainmesh = refine(domainmesh, redistribute=False)

    vCG = VectorFunctionSpace(domainmesh, hyperparameters["functionspace"], hyperparameters["functiondegree"])

    # controlfun.set_allow_extrapolation(True)
    # controlfun.rename("controlfun_coarse", "")
    # controlfun.vector().update_ghost_values()

    controlfun_fine = interpolate(controlfun, vCG)
    
    l2 = assemble((controlfun_fine - controlfun) ** 2 * dx(domain=controlfun.function_space().mesh()) )
    l2norm = assemble((controlfun) ** 2 * dx(domain=controlfun.function_space().mesh()) )

    print("L2 error", l2)
    print("rel L2 error", l2 / l2norm)
    print("L2 norm of control", l2norm)

    # controlfun.vector().update_ghost_values()



    print("Interpolated expression, writing...")

    with XDMFFile(hyperparameters["outputfolder"] + "/Interpolated_Control.xdmf") as xdmf:
        xdmf.write_checkpoint(controlfun_fine, "fine", 0.)

    controlFileInterpolated = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Interpolated_Control.hdf", "w")
    controlFileInterpolated.write(domainmesh, "mesh")
    controlFileInterpolated.write(controlfun_fine, "refined_control")
    controlFileInterpolated.close()

    return domainmesh, vCG, controlfun_fine

    
# def load_velocity(hyperparameters, controlfun):

#     assert os.path.isfile(hyperparameters["starting_guess"])

#     print("Will try to read starting guess")
    
#     print("max before loading", controlfun.vector()[:].max())

    
#     h5file = h5py.File(hyperparameters["starting_guess"])

#     print("keys in h5file", list(h5file.keys()))


#     if hyperparameters["interpolate"]:
#         print("--interoplate is set, trying to read coarse mesh")
        
#         coarse_mesh = Mesh()
#         hdf = HDF5File(coarse_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
#         hdf.read(coarse_mesh, '/mesh', False)
#         print("read mesh")

#         vCG_coarse = VectorFunctionSpace(coarse_mesh, "CG", 1)
#         controlfun_coarse = Function(vCG_coarse)

#         hdf = HDF5File(coarse_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
        
#         print("trying to read", hyperparameters["readname"])
#         hdf.read(controlfun_coarse, hyperparameters["readname"])

#         print("read coarse control, try to interpolate to fine mesh")

#         fmesh = controlfun.function_space().mesh()
#         print(controlfun)
#         print("controlfun_coarse", controlfun_coarse)
#         print(fmesh.coordinates().min(), fmesh.coordinates().max())
#         print("controlfun_coarse", coarse_mesh.coordinates().min(), coarse_mesh.coordinates().max())

#         controlfun = interpolate(controlfun_coarse, controlfun.function_space())

#     else:
#         print("trying to read velocity without loading mesh")
    
#         hdf = HDF5File(controlfun.function_space().mesh().mpi_comm(), hyperparameters["starting_guess"], 'r')
        
#         print("trying to read", hyperparameters["readname"])
#         hdf.read(controlfun, hyperparameters["readname"])

#     print("max after loading", controlfun.vector()[:].max())
#     print("Succesfully read starting guess")