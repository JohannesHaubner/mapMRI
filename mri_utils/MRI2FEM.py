import nibabel
from fenics import *
from fenics_adjoint import *
import numpy as np
from nibabel.affines import apply_affine

def read_image(filename, data_filter=None, storepath=None):
    
    print("Loading", filename)
    
    image2 = nibabel.load(filename)
    data = image2.get_fdata()


    # x0 = Point(0.0, 0.0, 0.0)
    # y0 = x0
    # z0 = y0

    # x1 = Point(data.shape[0], data.shape[1], data.shape[2])
    # y1 = x1
    # z1 = y1

    nx = data.shape[0] 
    ny = data.shape[1]
    nz = data.shape[2]

    assert nx == ny
    assert ny == nz

    # mesh = BoxMesh(MPI.comm_world, x0, y0, z0, x1, y1, z1, nx, ny, nz)
    # nx += 2
    mesh = UnitCubeMesh(nx, nx, nx)

    xyzm = mesh.coordinates()[:]

    # breakpoint()

    mesh.coordinates()[:] *= (nx - 1)

    # File("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/mesh.pvd") << mesh

    print(len(range(int(mesh.coordinates()[:].min()), int(mesh.coordinates()[:].max()))))

    # space = VectorFunctionSpace(mesh, "DG", 0, 1)
    space = FunctionSpace(mesh, "DG", 1)

    u_data = Function(space)

    try:
        ras2vox = image2.header.get_ras2vox()
        assert np.unique(ras2vox).size == 2
        assert 1. in np.unique(ras2vox)
        assert 0. in np.unique(ras2vox)
    except:
        raise NotImplementedError()
        print("image2.header.get_ras2vox()", ras2vox)

        ras2vox = np.linalg.inv(image2.header.get_vox2ras_tkr())

        print("image2.header.get_vox2ras_tkr() = ", ras2vox)

    xyz = space.tabulate_dof_coordinates().transpose()

    i, j, k = np.rint(xyz).astype("int")

    # breakpoint()
    # ijk = apply_affine(ras2vox, xyz).T
    # i, j, k = np.rint(ijk).astype("int")

    # breakpoint()
    u_data.vector()[:] = data[i, j, k]

    # breakpoint()

    # ijk = apply_affine(ras2vox, xyz).T
    # i, j, k = np.rint(ijk).astype("int")

    # print(i.min(), i.max())
    
    # if data_filter is not None:
    #     data = data_filter(data, ijk, i, j, k)
    #     u_data.vector()[:] = data[i, j, k]
    # else:
    #     print("No filter used, setting", np.where(np.isnan(data[i, j, k]), 1, 0).sum(), "/", i.size, " nan voxels to 0")
    #     data[i, j, k] = np.where(np.isnan(data[i, j, k]), 0, data[i, j, k])
    #     u_data.vector()[:] = data[i, j, k]

    # File("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/u_data.pvd") << u_data

    return mesh, u_data, 1


if __name__ == "__main__":

    testimg = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/input.mgz"
    
    read_image(testimg)

