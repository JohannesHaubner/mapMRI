from fenics import *
from dgregister.MRI2FEM import read_image, fem2mri
import nibabel
import numpy as np

PATH = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/coarsecropped/"
FName = PATH + "coarsenedernie_brain.mgz"

# PATH = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/"
# FName = PATH + "input.mgz"

domainmesh, Img, N1 = read_image(hyperparameters={"image": FName, "state_functionspace": "DG", "state_functiondegree":1}, 
                name="image", mesh=None, normalize=False)


image = nibabel.load(FName)

retimage = fem2mri(function=Img, imagepath=FName)

assert np.sum(np.isnan(retimage)) == 0

# OUTNAME=FName.replace(".mgz", "loaded_and_backprojected.mgz")
# nibabel.save(nibabel.Nifti1Image(retimage, affine=image.affine), OUTNAME)

# from mpi4py import MPI
# amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
# # amode = MPI.mode_wronly|MPI.mode_create
# comm = MPI.COMM_WORLD

# OUTNAME=FName.replace(".mgz", "loaded_and_backprojected.npy")
# fh = MPI.File.Open(comm, OUTNAME, amode)

# buffer = np.zeros_like(image.get_fdata())
# buffer[:] = comm.Get_rank()

# offset = comm.Get_rank()*buffer.nbytes
# fh.Write_at_all(offset, buffer)

# fh.Close()

# if comm.Get_rank() == 0:
#     # set_log_level(PROGRESS)
#     print(OUTNAME)
#     img = np.load(OUTNAME)
#     OUTNAME=FName.replace(".npy", ".mgz")
#     nibabel.save(nibabel.Nifti1Image(img, affine=image.affine), OUTNAME)
#     print("freeview ", FName, OUTNAME)

# else:
#     pass

