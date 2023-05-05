import nibabel
import numpy as np
import os
from fenics import *

import dgregister.config as config
MaxIter = 100

DeltaT = 1 / MaxIter
roiname = "input"
hyperparameters = {"optimize": False, "state_functiondegree": 1, "state_functionspace":"DG", "input": roiname + ".mgz", "timestepping":"explicitEuler"}
hyperparameters["preconditioner"] = "amg"

config.hyperparameters = hyperparameters

if __name__ == "__main__":


    from dgregister.MRI2FEM import read_image, fem2mri
    from dgregister.DGTransport import DGTransport

    os.chdir("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d")

    path = "./"



    inputimage = nibabel.load(roiname + ".mgz")
    aff =inputimage.affine
    inputimage = inputimage.get_fdata()



    (domainmesh, Img, NumData) = read_image(hyperparameters, name="input", mesh=None, 
            normalize=True, filter=False)

    vCG = VectorFunctionSpace(domainmesh, "CG", 1)

    velocity = Function(vCG)

    # Define boundary condition
    u_D = Expression(['0', '0', '0'], degree=0)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(vCG, u_D, boundary)


    velocity.vector()[:] = 3 # 10 / MaxIter


    bc.apply(velocity.vector())


    Img_deformed = DGTransport(Img, Wind=velocity, MaxIter=MaxIter, DeltaT=DeltaT, hyperparameters=hyperparameters, 
                                MassConservation=False, StoreHistory=False, FNameOut="",
                                    solver="krylov", timestepping=hyperparameters["timestepping"])


    hdf = HDF5File(domainmesh.mpi_comm(), "NewTarget.hdf", "w")
    hdf.write(domainmesh, "mesh")
    hdf.write(Img_deformed, "Target")
    hdf.write(Img, "Input")
    hdf.write(velocity, "velocity")
    hdf.close()

    Img = fem2mri(function=Img_deformed, shape=inputimage.shape)

    nibabel.save(nibabel.Nifti1Image(Img, aff), "newtarget.mgz")