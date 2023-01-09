import json
import os
import pathlib

import nibabel
import numpy
import numpy as np
from fenics import *
from fenics_adjoint import *
from nibabel.affines import apply_affine
from dgregister.helpers import get_largest_box, pad_with, cut_to_box, get_bounding_box, get_lumped_mass_matrices
from dgregister import find_velocity_ocd



def store2mgz(imgfile1, imgfile2, ijk1, outfolder):
    assert os.path.isdir(outfolder)

    data = np.zeros((256, 256, 256))

    for idx, img in enumerate([imgfile1, imgfile2]):
        image1 = nibabel.load(img)
        
         
        i1, j1, k1 = numpy.rint(ijk1).astype("int")

        data[i1, j1, k1] = 1.

        assert data.sum() > 100

        nii = nibabel.Nifti1Image(data, image1.affine)
        # breakpoint()
        nibabel.save(nii, outfolder + pathlib.Path(imgfile1).name.replace(".mgz", "affine" + str(idx) + "_meshindices.mgz"))

# from IPython import embed

def map_mesh(xmlfile1, imgfile1, imgfile2, mapping, box=None, coarsening_factor=2, outfolder=None):

    comm = MPI.comm_world
    nprocs = comm.Get_size()

    if nprocs != 1:
        raise NotImplementedError

    if int(coarsening_factor) not in [1, 2]:
        raise ValueError

    if xmlfile1.endswith(".xml"):
        brainmesh = Mesh(xmlfile1)
    else:
        brainmesh = Mesh()
        hdf = HDF5File(brainmesh.mpi_comm(), xmlfile1, "r")
        hdf.read(brainmesh, "/mesh", False)

    image1 = nibabel.load(imgfile1)

    ras2vox_tkr_inv1 = numpy.linalg.inv(image1.header.get_vox2ras_tkr())
    ras2vox1 = ras2vox_tkr_inv1

    xyz1 = brainmesh.coordinates()

    ijk1 = apply_affine(ras2vox1, xyz1).T
    # i1, j1, k1 = numpy.rint(ijk1).astype("int")

    store2mgz(imgfile1, imgfile2, ijk1, outfolder)

    i1, j1, k1 = ijk1

    print("Maximum values of indices before transform", i1.max(), j1.max(), k1.max())

    if box is not None:
    
        bounds = get_bounding_box(box)

        dxyz = [bounds[x].start for x in range(3)]
        # TODO FIXME is this correct ? 
    else:

        dxyz = [0, 0, 0]

    npad = 4 #  + 1

    ijk2 = []
    for i,j, k in zip(i1, j1, k1):
        i = (i + npad - dxyz[0]) / coarsening_factor
        j = (j + npad - dxyz[1]) / coarsening_factor
        k = (k + npad - dxyz[2]) / coarsening_factor
        
        point = (i,j,k)
        try:
            i2, j2, k2 = mapping(point)
        except RuntimeError:
            print(point, "not in BoxMesh")
            exit()

        if max([i2, j2, k2]) > 256:
            point = (i,j,k)
            print(point, "-->", (i2, j2, k2))
            exit()
        
        ijk2.append([i2, j2, k2])


    print("Maximum values of indices after transform", np.max(ijk2, axis=0))

    if np.max(ijk2) > 255:
        raise ValueError



    image2 = nibabel.load(imgfile2)

    # ras2vox2 = image2.header.get_ras2vox()
    vox2ras2 = image2.header.get_vox2ras_tkr()
    # ras2vox_tkr_inv2 = numpy.linalg.inv(vox2ras2)





    ijk2 = np.array(ijk2)

    print(ijk2.dtype, ijk2.shape)

    xyz2 = apply_affine(vox2ras2, ijk2)

    if xmlfile1.endswith(".xml"):
        brainmesh2 = Mesh(xmlfile1)
    else:
        brainmesh2 = Mesh()
        hdf = HDF5File(brainmesh2.mpi_comm(), xmlfile1, "r")
        hdf.read(brainmesh2, "/mesh", False)


    brainmesh2.coordinates()[:] = xyz2

    return brainmesh2



def make_mapping(cubemesh, v, jobfile, hyperparameters, ocd):


    mappings = []

    for coordinate in ["x[0]", "x[1]", "x[2]"]:

        # if os.path.isfile(jobfile + "postprocessing/" + coordinate + ".hdf"):

        #     coordinate_mapping = Function(V1)

        #     hdf = HDF5File(cubemesh.mpi_comm(), jobfile + "postprocessing/" + coordinate + ".hdf", "r")
        #     hdf.read(coordinate_mapping, "out")
        #     hdf.close()

        #     assert norm(coordinate_mapping) != 0

        #     mappings.append(coordinate_mapping)

        # else:

        

        # print("Inverting velocity for backtransport")
        # v.vector()[:] *= (-1)

        assert norm(v) > 0

        print("Transporting, ", coordinate, "coordinate")

        if ocd:
            V1 = FunctionSpace(cubemesh, "CG", 1)

            
            unity2 = Function(V1) #¤ cubeimg.function_space())
            unity2.vector()[:] = 0


            xin = interpolate(Expression(coordinate, degree=1), V1) # cubeimg.function_space())
            xout, _ = find_velocity_ocd.find_velocity(Img=xin, Img_goal=unity2, hyperparameters=hyperparameters, phi_eval=-v, projection=False)
        else:
            import dgregister.config as config
            
            config.hyperparameters = hyperparameters
            from dgregister.DGTransport import Transport


            V1 = FunctionSpace(cubemesh, hyperparameters["state_functionspace"], hyperparameters["state_functiondegree"])
            xin = interpolate(Expression(coordinate, degree=1), V1) # cubeimg.function_space())
            
            if hyperparameters["smoothen"]:
                raise NotImplementedError
                _, M_lumped_inv = get_lumped_mass_matrices(vCG=vCG)
            else:
                M_lumped_inv = None
            print("Calling Transport()")

            xout = Transport(Img=xin, Wind=-v, hyperparameters=hyperparameters,
                            MaxIter=hyperparameters["max_timesteps"], DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                            solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"])
            # find_velocity.find_velocity(Img=xin, Img_goal=unity2, hyperparameters=hyperparameters, phi_eval=-v, files=[], projection=False)
            # find_velocity(Img=Img, Img_goal=Img_goal, vCG=vCG, M_lumped_inv=M_lumped_inv, hyperparameters=hyperparameters, files=files, starting_guess=controlfun)
            V1_CG = FunctionSpace(cubemesh, "CG", 1)

            xout = project(xout, V1_CG)
            print("Projected xout to CG1")


        assert norm(xout) != 0

        mappings.append(xout)

        assert os.path.isdir(jobfile + "postprocessing/")

        hdf = HDF5File(cubemesh.mpi_comm(), jobfile + "postprocessing/" + coordinate + ".hdf", "w")
        hdf.write(xin, "in")
        hdf.write(xout, "out")
        hdf.close()

        with XDMFFile(jobfile + "postprocessing/" + coordinate + "_in.xdmf") as xdmf:
            xdmf.write_checkpoint(xin, "xin", 0.)


        with XDMFFile(jobfile + "postprocessing/" + coordinate + "_out.xdmf") as xdmf:
            xdmf.write_checkpoint(xout, "xout", 0.)

    assert len(mappings) == 3

    assert True not in [norm(x) == 0 for x in mappings]

    # for coordinate in ["x[0]", "x[1]", "x[2]"]:
    vxyz = Function(v.function_space())

    arr = np.zeros_like(vxyz.vector()[:])

    arr[0::3] = mappings[0].vector()[:]
    arr[1::3] = mappings[1].vector()[:]
    arr[2::3] = mappings[2].vector()[:]

    assert np.sum(arr) != 0

    vxyz.vector()[:] = arr
    
    assert norm(vxyz) != 0

    return vxyz