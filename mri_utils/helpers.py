from dolfin import *
from dolfin_adjoint import *
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
    
    print("max before loading", controlfun.vector()[:].max())

    
    h5file = h5py.File(hyperparameters["starting_guess"])

    print("keys in h5file", list(h5file.keys()))


    print("trying to read velocity without loading mesh")

    hdf = HDF5File(controlfun.function_space().mesh().mpi_comm(), hyperparameters["starting_guess"], 'r')
    
    print("trying to read", hyperparameters["readname"])
    hdf.read(controlfun, hyperparameters["readname"])

    print("max after loading", controlfun.vector()[:].max())
    print("Succesfully read starting guess")






def interpolate_velocity(hyperparameters, controlfun):

    assert os.path.isfile(hyperparameters["starting_guess"])

    print("Will try to read starting guess")
    
    print("max before loading", controlfun.vector()[:].max())

    
    h5file = h5py.File(hyperparameters["starting_guess"])

    print("keys in h5file", list(h5file.keys()))


    print("--interoplate is set, trying to read coarse mesh")
    
    coarse_mesh = Mesh()
    hdf = HDF5File(coarse_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
    hdf.read(coarse_mesh, '/mesh', False)
    print("read mesh")

    vCG_coarse = VectorFunctionSpace(coarse_mesh, "CG", 1)
    controlfun_coarse = Function(vCG_coarse)

    hdf = HDF5File(coarse_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
    
    print("trying to read", hyperparameters["readname"])
    hdf.read(controlfun_coarse, hyperparameters["readname"])

    print("read coarse control, try to interpolate to fine mesh")

    fmesh = controlfun.function_space().mesh()
    print(controlfun)
    print("controlfun_coarse", controlfun_coarse)
    print(fmesh.coordinates().min(), fmesh.coordinates().max())
    print("controlfun_coarse", coarse_mesh.coordinates().min(), coarse_mesh.coordinates().max())

    fun = Expression(("f[0]", "f[1]", "f[2]"), f=controlfun_coarse, degree=1)

    print("will try to interpolate Expression")
    controlfun = interpolate(fun, controlfun.function_space())

    print("Interpolated expression, writing...")

    controlFileInterpolated = HDF5File(fmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Interpolated_Control.hdf", "w")
    controlFileInterpolated.write(fmesh, "mesh")
    controlFileInterpolated.write(controlfun, "control")
    controlFileInterpolated.close()

    print("Wrote control, exiting")
    exit()

    
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