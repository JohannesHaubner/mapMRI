from fenics import *
from fenics_adjoint import *
import nibabel
import numpy as np
from nibabel.affines import apply_affine
import pathlib


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass


def read_image(hyperparameters, name, mesh=None, printout=True, normalize=True):
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

    hyperparameters[name + ".shape"] = list(data.shape)
    if printout:
        print_overloaded("dimension of image:", data.shape, "(", data.size, "voxels)")

    nx = data.shape[0] 
    ny = data.shape[1]
    nz = data.shape[2]

    # The dof coordinates for DG0 are in the middle of the cell. 
    # Shift everything by -0.5 so that the rounded dof coordinate corresponds to the voxel idx.
    dx = 0.5 
    
    if mesh is None:
        if nz == 1:

            mesh = RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(nx, ny), nx, ny)


            print_overloaded("Created rectangle mesh")
        else:
            mesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)
            print_overloaded("Created box mesh")

    space = FunctionSpace(mesh, hyperparameters["state_functionspace"], 0) #hyperparameters["state_functiondegree"])

    u_data = Function(space)

    xyz = space.tabulate_dof_coordinates().transpose()

    if nz == 1:
        xyz = np.stack((xyz[0, :], xyz[1, :], np.zeros_like(xyz[0, :])), axis=0)



    # The dof coordinates for DG0 are in the middle of the cell. 
    # Shift everything by -0.5 so that the rounded dof coordinate corresponds to the voxel idx.
    i, j, k = np.rint(xyz - dx).astype("int")

    if normalize:
        print_overloaded("Normalizing data")
        print_overloaded("data.max()", data.max())
        data /= data.max()

    if np.min(data) < 0:
        
        mask = np.where(data < 0, True, False)
        print_overloaded("-"*80)
        print_overloaded("Warning:", mask.sum(), "/", mask.size, "voxels < 0 in", name)
        print_overloaded("Smallest value", np.min(data), "largest value:", np.max(data))
        print_overloaded("Will apply ReLU to image (threshold negative values to 0)")
        hyperparameters["smallest_voxel_value"] = np.min(data)

        print_overloaded("-"*80)
        
        data = np.where(data < 0, 0, data)

        


    u_data.vector()[:] = data[i, j, k]


    space = FunctionSpace(mesh, hyperparameters["state_functionspace"], hyperparameters["state_functiondegree"])
    
    # if "outputfolder" in hyperparameters.keys():
    #     with XDMFFile(hyperparameters["outputfolder"] + "/" + str(pathlib.Path(hyperparameters[name]).stem) +"_DG0.xdmf") as xdmf:
    #         xdmf.write_checkpoint(u_data, "ImgDG0", 0.)

    u_data = project(u_data, space)


    return mesh, u_data, 1  

if __name__ == "__main__":

    testimg = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/input.mgz"
    
    read_image(testimg)

