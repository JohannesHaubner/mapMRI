from fenics import *
import numpy as np
import numpy
import nibabel
import os, pathlib
import nibabel
from nibabel.affines import apply_affine

hdf5file = None
parcfile = None
    # ras2vox = image2.header.get_ras2vox()

    # ras2vox_tkr_inv = numpy.linalg.inv(image2.header.get_vox2ras_tkr())
    # # if tkr is True:
    # #     ras2vox = ras2vox_tkr_inv
    # ras2vox = ras2vox_tkr_inv

    # xyz = space.tabulate_dof_coordinates()
    # ijk = apply_affine(ras2vox, xyz).T
    # i, j, k = numpy.rint(ijk).astype("int")


    # Read the mesh and mesh data from .h5:
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), hdf5file, "r")
hdf.read(mesh, "/mesh", False)  

d = mesh.topology().dim()
subdomains = MeshFunction("size_t", mesh, d)
hdf.read(subdomains, "/subdomains")
boundaries = MeshFunction("size_t", mesh, d-1)
hdf.read(boundaries, "/boundaries")
hdf.close()

# Load parcellation image and data 
image = nibabel.load(parcfile)
data = image.get_fdata() 

# Find the transformation to T1 voxel space from 
# surface RAS (aka mesh) coordinates 
vox2ras = image.header.get_vox2ras_tkr()
ras2vox = numpy.linalg.inv(vox2ras)

# Extract RAS coordinates of cell midpoints
xyz = numpy.array([cell.midpoint()[:]
                for cell in cells(mesh)])

## This version is equivalent, and faster, more extendable to other
## spaces, but requires more background knowledge.
#DG0 = FunctionSpace(mesh, "DG", 0)
#imap = DG0.dofmap().index_map()
#num_dofs_local =  imap.local_range()[1]-imap.local_range()[0]
#xyz = DG0.tabulate_dof_coordinates().reshape((num_dofs_local,-1))

# Convert to voxel space and voxel indices: for cell c,
# i[c], j[c], k[c] give the corresponding voxel indices.
abc = apply_affine(ras2vox, xyz).T  
ijk = numpy.rint(abc).astype("int")  
(i, j, k) = ijk