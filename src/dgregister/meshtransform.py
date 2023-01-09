import json
import os
import pathlib

import nibabel
import numpy
import numpy as np
from fenics import *
from fenics_adjoint import *
from nibabel.affines import apply_affine
from dgregister import MRI2FEM, DGTransport, find_velocity_ocd, mesh_tools
from mask_mri import get_bounding_box

# from IPython import embed

def map_meshes(xmlfile1, imgfile1, imgfile2, box=None):


    args = {"mesh": xmlfile1}

    if args["mesh"].endswith(".xml"):
        brainmesh = Mesh(args["mesh"])
    else:
        brainmesh = Mesh()
        hdf = HDF5File(brainmesh.mpi_comm(), args["mesh"], "r")
        hdf.read(brainmesh, "/mesh", False)


    image1 = nibabel.load(imgfile1)

    ras2vox_tkr_inv1 = numpy.linalg.inv(image1.header.get_vox2ras_tkr())
    ras2vox1 = ras2vox_tkr_inv1

    xyz1 = brainmesh.coordinates()

    ijk1 = apply_affine(ras2vox1, xyz1).T
    i1, j1, k1 = numpy.rint(ijk1).astype("int")

    if box is not None:
    
    bounds = get_bounding_box(box)

    dxyz = [bounds[x].start for x in range(3)]
    # TODO FIXME is this correct ? 

    ijk2 = []
    for i,j, k in zip(i1, j1, k1):
        i = (i - dxyz[0]) / 2
        j = (j - dxyz[1]) / 2
        k = (k - dxyz[2]) / 2
        
        i2, j2, k2 = mapping((i,j,k))
        ijk2.append([i2, j2, k2])

    image2 = nibabel.load(imgfile2)

    # ras2vox2 = image2.header.get_ras2vox()
    vox2ras2 = image2.header.get_vox2ras_tkr()
    # ras2vox_tkr_inv2 = numpy.linalg.inv(vox2ras2)

    ijk2 = np.array(ijk2)

    print(ijk2.dtype, ijk2.shape)

    xyz2 = apply_affine(vox2ras2, ijk2)

    if args["mesh"].endswith(".xml"):
        brainmesh2 = Mesh(args["mesh"])
    else:
        brainmesh2 = Mesh()
        hdf = HDF5File(brainmesh2.mpi_comm(), args["mesh"], "r")
        hdf.read(brainmesh2, "/mesh", False)

    # embed()

    brainmesh2.coordinates()[:] = xyz2
