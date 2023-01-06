from fenics import *
from dgregister.MRI2FEM import read_image, fem2mri
import nibabel
import numpy as np
import os

if "/home/bastian" in os.getcwd():
    
    PATH = "/home/bastian/D1/registration/mri2fem-data/processed/coarsecropped/"
    FName = PATH + "coarsenedernie_brain.mgz"
else:
    PATH = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/coarsecropped/"
    FName = PATH + "coarsenedernie_brain.mgz"

    # PATH = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/"
    # FName = PATH + "input.mgz"

    # PATH = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/"
    # FName = PATH + "mpi_testimage.mgz"

domainmesh, Img, _ = read_image(hyperparameters={"image": FName, "state_functionspace": "DG", "state_functiondegree":1}, 
                name="image", mesh=None, normalize=False, threshold=False)


image = nibabel.load(FName)
affine = image.affine
image = image.get_fdata()
shape = nibabel.load(FName).get_fdata().shape

retimage = fem2mri(function=Img, shape=shape)


if MPI.comm_world.rank == 0:

    reldiff = np.mean(np.abs(image-retimage)) / np.mean(np.abs(image))

    print("np.sum(retimage)=", np.sum(retimage))
    print("np.sum(image)   =", np.sum(image))

    print("np.max(retimage)=", np.max(retimage))
    print("np.max(image)   =", np.max(image))

    print("np.min(retimage)=", np.min(retimage))
    print("np.min(image)   =", np.min(image))

    OUTNAME=FName.replace(".mgz", "loaded_and_backprojected.mgz")
    nibabel.save(nibabel.Nifti1Image(retimage, affine=affine), OUTNAME)


    print("Stored image, to view the result run")
    print("freeview ", FName, OUTNAME)

    assert reldiff < 1e-15, "Reldiff is =" + str(reldiff)
