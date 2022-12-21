from fenics import *
from fenics_adjoint import *
import nibabel
import numpy as np
from nibabel.affines import apply_affine

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

    

def read_image(hyperparameters, name, mesh=None, storepath=None, printout=True, normalize=True):
    if printout:
        print_overloaded("Loading", hyperparameters[name])
    
    if hyperparameters[name].endswith(".mgz"):
        image2 = nibabel.load(hyperparameters[name])
        data = image2.get_fdata()
    elif hyperparameters[name].endswith(".png"):
        from PIL import Image
        img = Image.open(hyperparameters[name])
        img = img.convert("L")
        data = np.array(img)

        data = np.expand_dims(data, -1)
        # pass

    hyperparameters[name + ".shape"] = list(data.shape)
    if printout:
        print_overloaded("dimension of image:", data.shape, "(", data.size, "voxels)")

    # x0 = Point(0.0, 0.0, 0.0)
    # y0 = x0
    # z0 = y0

    # x1 = Point(data.shape[0], data.shape[1], data.shape[2])
    # y1 = x1
    # z1 = y1

    nx = data.shape[0] 
    ny = data.shape[1]
    nz = data.shape[2]



    # assert nx == ny
    # assert ny == nz

    # mesh = BoxMesh(MPI.comm_world, x0, y0, z0, x1, y1, z1, nx, ny, nz)
    # nx += 2
    if mesh is None:
        if nz == 1:
            mesh = UnitSquareMesh(MPI.comm_world, nx, ny)
        else:
            mesh = UnitCubeMesh(MPI.comm_world, nx, ny, nz)

    space = FunctionSpace(mesh, hyperparameters["state_functionspace"], hyperparameters["state_functiondegree"])

    u_data = Function(space)

    # try:
    #     ras2vox = image2.header.get_ras2vox()
    #     assert np.unique(ras2vox).size == 2
    #     assert 1. in np.unique(ras2vox)
    #     assert 0. in np.unique(ras2vox)
    # except:
    #     # pass
    #     # raise NotImplementedError()
    #     print_overloaded("image2.header.get_ras2vox()", ras2vox)

    #     ras2vox = np.linalg.inv(image2.header.get_vox2ras_tkr())

    #     print_overloaded("image2.header.get_vox2ras_tkr() = ", ras2vox)

    #     del ras2vox

    xyz = space.tabulate_dof_coordinates().transpose()

    xyz[0, :] *= (nx - 1)
    xyz[1, :] *= (ny - 1)
    if nz > 1:

        xyz[2, :] *= (nz - 1)

    else:

        xyz2 = np.zeros((3, xyz.shape[1]))

        xyz2[0, :] = xyz[0, :]
        xyz2[1, :] = xyz[1, :]

        xyz = xyz2
        del xyz2

    i, j, k = np.rint(xyz).astype("int")

    
    # ijk = apply_affine(ras2vox, xyz).T
    # i, j, k = np.rint(ijk).astype("int")

    # breakpoint()
    u_data.vector()[:] = data[i, j, k]



    if normalize:
        if printout:
            print_overloaded("Normalizing image")
            print_overloaded("Img.vector()[:].max()", u_data.vector()[:].max())

        u_data.vector()[:] *= 1 / u_data.vector()[:].max()
        # Img_goal.vector()[:] *= 1 / Img_goal.vector()[:].max()

        if printout:
            print_overloaded("Applying ReLU() to image")
        
        u_data.vector()[:] = np.where(u_data.vector()[:] < 0, 0, u_data.vector()[:])
        # Img_goal.vector()[:] = np.where(Img_goal.vector()[:] < 0, 0, Img_goal.vector()[:])

    return mesh, u_data, 1


if __name__ == "__main__":

    testimg = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/input.mgz"
    
    read_image(testimg)

