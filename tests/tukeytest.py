from fenics import *
from fenics_adjoint import *
import numpy as np
import nibabel
from ufl import sign
import os
from dgregister.MRI2FEM import read_image
from dgregister.tukey import *


def ReLU(x):
    return (x + abs(x)) / 2

if __name__ == "__main__":

    path = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/"

    os.chdir(path)

    imgpath1="affine_registered/cropped_abby_brain.mgz"
    imgpath2="affine_registered/cropped_ernie_brain.mgz"

    imgpath1="coarsecropped/coarsenedernie_brain.mgz"
    imgpath2="coarsecropped/coarsenedabbytoernie.mgz"

    im1 = ReLU(nibabel.load(imgpath1).get_fdata())
    im2 = ReLU(nibabel.load(imgpath2).get_fdata())
    print("sigma, computed by median", sigma(im1-im2))
    # print("sigma, computed by median", sigma2(im1-im2))

    # print(np.mean(im1) / np.mean(im2))

    # exit()

    imgmesh, Img1, _ = read_image(hyperparameters={"input": imgpath1}, name="input", mesh=None, printout=True, threshold=True, 
                    state_functionspace="DG", state_functiondegree=0, 
                    normalize=False, filter=False)

    _, Img2, _ = read_image(hyperparameters={"input": imgpath2}, name="input", mesh=imgmesh, printout=True, threshold=True, 
                    state_functionspace="DG", state_functiondegree=0, 
                    normalize=False, filter=False)


    

    # imgmesh = UnitSquareMesh(2,2)
    # V = FunctionSpace(imgmesh, "DG", 1)
    # Img1 = interpolate(Expression("x[0] + 1.5", degree=1), V)
    # Img2 = interpolate(Expression("x[0] + 0.5", degree=1), V)

    dx = dx(imgmesh)

    vol = assemble(1*dx)

    # print("sigma, estimated by mean", sigma(Img1-Img2, vol=vol, dx=dx))

    print(assemble(Img1*dx) / assemble(Img2*dx))
    print(assemble(Img2*dx) / assemble(Img1*dx))


    # print("tukeymean", tukeyloss(Img1, Img2, vol=vol, dx=dx) / vol)
    # print("tukeymean_np", np.mean(tukeyloss(im1, im2, vol=None)))
    # print("l2mean", l2loss(Img1, Img2, vol=vol, dx=dx) / vol)
    # print("l2loss_np", np.mean(l2loss(im1, im2, vol=None)))