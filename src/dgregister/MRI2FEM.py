from fenics import *
from fenics_adjoint import *
import nibabel
import numpy as np
from nibabel.affines import apply_affine
import pathlib

# from mpi4py import MPI

comm = MPI.comm_world
nprocs = comm.Get_size()

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

# The dof coordinates for DG0 are in the middle of the cell. 
# Shift everything by -0.5 so that the rounded dof coordinate corresponds to the voxel idx.
dxyz = 0.5 


def fem2mri(function: Function, shape) -> np.ndarray:
    
    V0 = FunctionSpace(function.function_space().mesh(), "DG", 0)

    function0 = project(function, V0)

    dof_values = function0.vector()[:]
    coordinates = V0.tabulate_dof_coordinates()

    gathered_coordinates = MPI.comm_world.gather(coordinates, root=0)
    gathered_values = MPI.comm_world.gather(dof_values, root=0)

    if MPI.comm_world.rank == 0:
        xy = np.vstack(gathered_coordinates)

        vals = np.hstack(gathered_values)

        dxyz = 0.5
        ijk = np.rint(xy - dxyz).astype("int")

        i = ijk[:, 0]
        j = ijk[:, 1]
        k = ijk[:, 2]
        
        retimage = np.zeros(shape=shape) + np.nan

        vals = np.squeeze(vals)

        assert np.isnan(vals).sum() == 0, "nan in fenics function after gathering?"

        retimage[i, j, k] = vals


        return retimage


def read_image(hyperparameters, name, mesh=None, printout=True, threshold=True, normalize=True):
    
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
    
    if mesh is None:
        if nz == 1:

            mesh = RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(nx, ny), nx, ny)
            print_overloaded("Created rectangle mesh")
        else:
            mesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)
            print_overloaded("Created box mesh")

    space = FunctionSpace(mesh, hyperparameters["state_functionspace"], 0) #hyperparameters["state_functiondegree"])

    u_data = Function(space)

    xyz = space.tabulate_dof_coordinates()
    
    # xyz = xyz.transpose()
    # if nz == 1:
    #     xyz = np.stack((xyz[0, :], xyz[1, :], np.zeros_like(xyz[0, :])), axis=0)
    # # The dof coordinates for DG0 are in the middle of the cell. 
    # # Shift everything by -0.5 so that the rounded dof coordinate corresponds to the voxel idx.
    # i, j, k = np.rint(xyz - dxyz).astype("int")

    if nz == 1:
        raise NotImplementedError

    ijk = np.rint(xyz - dxyz).astype("int")
    i = ijk[:, 0]
    j = ijk[:, 1]
    k = ijk[:, 2]

    if normalize:
        print_overloaded("Normalizing data")
        print_overloaded("data.max()", data.max())
        data /= data.max()

    if threshold and np.min(data) < 0:
        
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
    
    u_data = project(u_data, space)

    return mesh, u_data, 1  

if __name__ == "__main__":

    testimg = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/input.mgz"
    
    read_image(testimg)

