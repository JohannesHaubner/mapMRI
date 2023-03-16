from fenics import *
from fenics_adjoint import *
from scipy import ndimage

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

import nibabel
import numpy as np


comm = MPI.comm_world
nprocs = comm.Get_size()


# The dof coordinates for DG0 are in the middle of the cell. 
# Shift everything by -0.5 so that the rounded dof coordinate corresponds to the voxel idx.
dxyz = 0.5 


def fem2mri(function, shape):
    
    V0 = FunctionSpace(function.function_space().mesh(), "DG", 0)

    function0 = project(function, V0)

    dof_values = function0.vector()[:]
    coordinates = V0.tabulate_dof_coordinates()

    gathered_coordinates = MPI.comm_world.gather(coordinates, root=0)
    gathered_values = MPI.comm_world.gather(dof_values, root=0)

    if np.max(shape) > 250:
        raise NotImplementedError("You are probably trying to map back to original size, this requires vox2ras")

    if MPI.rank(MPI.comm_world) == 0:
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



class Projector():
    def __init__(self, V):
        self.v = TestFunction(V)
        u = TrialFunction(V)
        form = inner(u, self.v)*dx
        self.A = assemble(form, annotate=False)
        # self.solver = LUSolver(self.A)
        self.V = V
        self.solver = KrylovSolver()
        self.solver.set_operators(self.A, self.A)
        print_overloaded("Solver in Projector: Krylov")
        # self.uh = Function(V)
    
    
    def project(self, f):
        L = inner(f, self.v)*dx
        b = assemble(L, annotate=False)
        
        uh = Function(self.V)
        self.solver.solve(uh.vector(), b)
        
        return uh


def read_image(filename, name, mesh=None, printout=True, threshold=True, projector=None,
                state_functionspace="DG", state_functiondegree=1, hyperparameters=None):
    
    if printout:
        print_overloaded("Loading", filename)
    
    if filename.endswith(".mgz"):
        image2 = nibabel.load(filename)
        data = image2.get_fdata()
    elif filename.endswith(".png"):
        from PIL import Image
        img = Image.open(filename)
        img = img.convert("L")
        data = np.array(img)

        data = np.expand_dims(data, -1)

    if "smoothen_image" in hyperparameters.keys() and hyperparameters["smoothen_image"]:
        sigma = 0.5
        print_overloaded("Applying Gauss filter to image, sigma=", sigma)
        data = ndimage.gaussian_filter(data, sigma=sigma)

    if hyperparameters is not None:
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

    space = FunctionSpace(mesh, "DG", 0)

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
    if nz != 1:
        k = ijk[:, 2]


    if threshold and np.min(data) < 0:
        
        mask = np.where(data < 0, True, False)
        print_overloaded("-"*80)
        print_overloaded("Warning:", mask.sum(), "/", mask.size, "voxels < 0 in", name)
        print_overloaded("Smallest value", np.min(data), "largest value:", np.max(data))
        print_overloaded("Will apply ReLU to image (threshold negative values to 0)")
        if hyperparameters is not None:
            hyperparameters["smallest_voxel_value"] = np.min(data)

        print_overloaded("-"*80)
        
        data = np.where(data < 0, 0, data)

    print_overloaded(name, "mean intensity", np.mean(data))
    print_overloaded(name, "median intensity", np.median(data))
    print_overloaded(name, "min intensity", np.min(data))
    print_overloaded(name, "max intensity", np.max(data))

    if nz != 1:
        u_data.vector()[:] = data[i, j, k]
    else:
        u_data.vector()[:] = data[i, j]

    if state_functiondegree > 0:

        if projector is None:
            projector = Projector(FunctionSpace(mesh, state_functionspace, state_functiondegree))
            print_overloaded("Initialized projector")
        else:
            print_overloaded("Reusing projector")
        
        u_data = projector.project(u_data)

    return mesh, u_data, np.max(data), projector

if __name__ == "__main__":

    testimg = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/input.mgz"
    
    read_image(testimg)

