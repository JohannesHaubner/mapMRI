import os
import numpy as np
import pathlib
from fenics import *
# from fenics_adjoint import *

from MRI2FEM import read_image
import json
import copy
parameters['ghost_mode'] = 'shared_facet'

resultpath = pathlib.Path("/home/bastian/D1/imageregistration_outputs")

# Everything after 100 iterations

results = {}

#  Alpha, Time Steps # Final F value, Projg
results[(1e-4, 11)] = (6.9698146314025123E-003, 3.666E-07)
results[(1e-4, 52)] = (6.9585702068614740E-003, 4.029E-07)
results[(1e-5, 52)] = (6.8147170928098905E-003, 3.651E-06)
results[(1e-5, )] = 0
results[(1e-6, )] = 0
results[(1e-7, 11)] = (6.7789178197843860E-003, 6.779E-03)

results_nosmoothen = {}
results_nosmoothen[(1e-4, 11)] = (5.8547268452063513E-003, 5.916E-04)


alphas = [1e-4, 1e-5, 1e-6, 1e-7]
buffers = [1, 0.2]

timestep_lookup = {1: 11, 0.2: 52}
foldername_lookup = {1: "", 0.2: "Buffer"}


for dt_buffer in buffers:

    for alpha in alphas:

        print()
        
        foldername = "createStartingGuess" + foldername_lookup[dt_buffer] + "A" + format(alpha, ".0e").replace("0", "")

        resultfolder = resultpath / foldername

        if "Finalstate.pvd" not in os.listdir(resultfolder):
            continue
        if "State.hdf" not in os.listdir(resultfolder):
            print(foldername, "has no State.hdf")
            continue
        
        
        hyperparameters = copy.deepcopy(json.load(open(resultfolder / "hyperparameters.json")))
        
        print(foldername)

        imagemesh = Mesh()
        hdf = HDF5File(imagemesh.mpi_comm(), str(resultfolder / "State.hdf"), 'r')

        hdf.read(imagemesh, "/mesh", False)

        V = FunctionSpace(imagemesh, "DG", 1)
        state = Function(V)
        state0 = Function(V)
        hdf.read(state, "-1")
        hdf.read(state0, "0")
        hdf.close()

        # hdf = HDF5File(imagemesh.mpi_comm(), str(resultfolder / "VelocityField.hdf"), 'r')
        # vCG = VectorFunctionSpace(imagemesh, hyperparameters["functionspace"], hyperparameters["functiondegree"])
        # v = Function(vCG)
        # hdf.read(v, "-1")
        # hdf.close()

        # print("max velocity", max(v.vector()[:]))
        # print("mean velocity", np.mean(v.vector()[:]))
        # reg = assemble(alpha*(v)**2*dx)
        # print("Regularization:", reg)

        (domainmesh, inputimage, NumData) = read_image(hyperparameters, name="input", mesh=None, printout=False)
        (_, targetimage, NumData) = read_image(hyperparameters, name="target", mesh=domainmesh, printout=False)

        breakpoint()

        # inputimage.vector()[:] *= 1 / inputimage.vector()[:].max()
        # targetimage.vector()[:] *= 1 / targetimage.vector()[:].max()

        dx = dx(domain=inputimage.function_space().mesh())

        d0 = assemble((targetimage - inputimage) ** 2 * dx)
        din = assemble((state - inputimage) ** 2 * dx)
        din0 = assemble((state0 - inputimage) ** 2 * dx)
        dstate = assemble((state - targetimage) ** 2 * dx)
        dstate0 = assemble((state - state0) ** 2 * dx)
    
        # print("state, inputimage, target boundary int:", assemble(state*ds), assemble(inputimage*ds), assemble(targetimage*ds))
        # print("assemble((state0 - state0) ** 2 * dx)", assemble((state0 - state0) ** 2 * dx))
        
        print("Error between input and target", d0 / 2)
        print("Error between input and state", din / 2)
        print("Error between state0 and input", din0 / 2)
        print("Error between state and state0", dstate0 / 2)
        print("Error between state and target", dstate / 2)
        try:
            print("Comparison to log: ", results[(alpha, timestep_lookup[dt_buffer])][0])
        except KeyError:
            pass


        exit()

        




