from fenics import *
from fenics_adjoint import *
import os
import h5py
import numpy as np

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

def get_lumped_mass_matrices(vCG):

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
    diaginv[:] = 1.0 / diag[:]
    M_lumped.set_diagonal(diag)
    M_lumped_inv.set_diagonal(diaginv)

    return M_lumped, M_lumped_inv


def load_velocity(hyperparameters, controlfun):

    assert os.path.isfile(hyperparameters["starting_guess"])

    print_overloaded("Will try to read starting guess")
    h5file = h5py.File(hyperparameters["starting_guess"])

    print_overloaded("keys in h5file", list(h5file.keys()))

    if controlfun is not None:
        print_overloaded("max before loading", controlfun.vector()[:].max())
        working_mesh = controlfun.function_space().mesh()
        print_overloaded("trying to read velocity without loading mesh")
        hdf = HDF5File(working_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')

        
    else:
        print_overloaded("Reading mesh")
        working_mesh = Mesh()
        hdf = HDF5File(working_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
        hdf.read(working_mesh, "/mesh", False)
        
        vCG = VectorFunctionSpace(working_mesh, hyperparameters["functionspace"], hyperparameters["functiondegree"])
        controlfun = Function(vCG)

    print_overloaded("trying to read", hyperparameters["readname"])
    hdf.read(controlfun, hyperparameters["readname"])
    hdf.close()

    print_overloaded("max after loading", controlfun.vector()[:].max())
    print_overloaded("Succesfully read starting guess")

    return working_mesh, vCG, controlfun





def interpolate_velocity(hyperparameters, domainmesh, vCG, controlfun, store_pvd=False):

    # controlFile = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Loaded_Control.hdf", "w")
    # controlFile.write(domainmesh, "mesh")
    # controlFile.write(controlfun, "loaded_control")
    # controlFile.close()

    # with XDMFFile(hyperparameters["outputfolder"] + "/Loaded_Control.xdmf") as xdmf:
    #     xdmf.write_checkpoint(controlfun, "coarse", 0.)
    if store_pvd:
        File(hyperparameters["outputfolder"] + "/Loaded_Control.pvd") << controlfun

    print_overloaded("parameters['ghost_mode'] in interpolate_velocity()", parameters['ghost_mode'])
    print_overloaded("trying to refine the mesh")

    domainmesh = refine(domainmesh, redistribute=False)

    vCG = VectorFunctionSpace(domainmesh, hyperparameters["functionspace"], hyperparameters["functiondegree"])

    controlfun_fine = interpolate(controlfun, vCG)

    # l2 = assemble((controlfun_fine - controlfun) ** 2 * dx(domain=controlfun.function_space().mesh()) )
    # l2norm = assemble((controlfun) ** 2 * dx(domain=controlfun.function_space().mesh()) )

    # print_overloaded("COARSE MESH")
    # print_overloaded("L2 error", l2)
    # print_overloaded("rel L2 error", l2 / l2norm)
    # print_overloaded("L2 norm of control", l2norm)

    print_overloaded("FINE MESH")
    
    l2 = assemble((controlfun_fine - controlfun) ** 2 * dx(domain=controlfun_fine.function_space().mesh()) )
    l2norm = assemble((controlfun) ** 2 * dx(domain=controlfun_fine.function_space().mesh()) )

    print_overloaded("L2 error", l2)
    print_overloaded("rel L2 error", l2 / l2norm)
    print_overloaded("L2 norm of control", l2norm)


    print_overloaded("Interpolated expression, writing...")

    # with XDMFFile(hyperparameters["outputfolder"] + "/Interpolated_Control.xdmf") as xdmf:
    #     xdmf.write_checkpoint(controlfun_fine, "fine", 0.)


    if store_pvd:
        File(hyperparameters["outputfolder"] + "/Interpolated_Control.pvd") << controlfun_fine

    controlFileInterpolated = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Interpolated_Control.hdf", "w")
    controlFileInterpolated.write(domainmesh, "mesh")
    controlFileInterpolated.write(controlfun_fine, "refined_control")
    controlFileInterpolated.close()

    if hyperparameters["debug"]:
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--debug is set, exiting here")
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--------------------------------------------------------------------------------------------------------")

        exit()

    return domainmesh, vCG, controlfun_fine

    
# def load_velocity(hyperparameters, controlfun):

#     assert os.path.isfile(hyperparameters["starting_guess"])

#     print_overloaded("Will try to read starting guess")
    
#     print_overloaded("max before loading", controlfun.vector()[:].max())

    
#     h5file = h5py.File(hyperparameters["starting_guess"])

#     print_overloaded("keys in h5file", list(h5file.keys()))


#     if hyperparameters["interpolate"]:
#         print_overloaded("--interoplate is set, trying to read coarse mesh")
        
#         coarse_mesh = Mesh()
#         hdf = HDF5File(coarse_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
#         hdf.read(coarse_mesh, '/mesh', False)
#         print_overloaded("read mesh")

#         vCG_coarse = VectorFunctionSpace(coarse_mesh, "CG", 1)
#         controlfun_coarse = Function(vCG_coarse)

#         hdf = HDF5File(coarse_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
        
#         print_overloaded("trying to read", hyperparameters["readname"])
#         hdf.read(controlfun_coarse, hyperparameters["readname"])

#         print_overloaded("read coarse control, try to interpolate to fine mesh")

#         fmesh = controlfun.function_space().mesh()
#         print_overloaded(controlfun)
#         print_overloaded("controlfun_coarse", controlfun_coarse)
#         print_overloaded(fmesh.coordinates().min(), fmesh.coordinates().max())
#         print_overloaded("controlfun_coarse", coarse_mesh.coordinates().min(), coarse_mesh.coordinates().max())

#         controlfun = interpolate(controlfun_coarse, controlfun.function_space())

#     else:
#         print_overloaded("trying to read velocity without loading mesh")
    
#         hdf = HDF5File(controlfun.function_space().mesh().mpi_comm(), hyperparameters["starting_guess"], 'r')
        
#         print_overloaded("trying to read", hyperparameters["readname"])
#         hdf.read(controlfun, hyperparameters["readname"])

#     print_overloaded("max after loading", controlfun.vector()[:].max())
#     print_overloaded("Succesfully read starting guess")