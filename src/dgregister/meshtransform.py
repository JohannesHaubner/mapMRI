import json
import os
import pathlib
from tqdm import tqdm
import nibabel
import numpy
import numpy as np
from fenics import *
from fenics_adjoint import *
from nibabel.affines import apply_affine
from dgregister.helpers import get_largest_box, pad_with, cut_to_box, get_bounding_box, get_lumped_mass_matrices
from dgregister import find_velocity_ocd

from IPython import embed

def store2mgz(imgfile1, imgfile2, ijk1, outfolder):
    assert os.path.isdir(outfolder)

    data = np.zeros((256, 256, 256))

    for idx, img in enumerate([imgfile1, imgfile2]):
        image1 = nibabel.load(img)
                 
        i1, j1, k1 = numpy.rint(ijk1).astype("int")

        data[i1, j1, k1] = 1.

        assert data.sum() > 100

        nii = nibabel.Nifti1Image(data, image1.affine)
        nibabel.save(nii, outfolder + pathlib.Path(imgfile1).name.replace(".mgz", "affine" + str(idx) + "_meshindices.mgz"))


def map_mesh(xmlfile1: str, imgfile1: str, imgfile2: str, mapping: Function,  
    coarsening_factor: int, npad: int, box: np.ndarray=None, outfolder=None):

    assert norm(mapping) != 0

    comm = MPI.comm_world
    nprocs = comm.Get_size()

    if nprocs != 1:
        raise NotImplementedError

    if coarsening_factor not in [1, 2]:
        raise ValueError

    if npad not in [0, 4]:
        raise ValueError

    if xmlfile1.endswith(".xml"):
        brainmesh = Mesh(xmlfile1)
    else:
        brainmesh = Mesh()
        hdf = HDF5File(brainmesh.mpi_comm(), xmlfile1, "r")
        hdf.read(brainmesh, "/mesh", False)

    image1 = nibabel.load(imgfile1)
    image2 = nibabel.load(imgfile2)

    tkr = True

    if tkr:
        print("Using trk RAS")
        ras2vox1 = numpy.linalg.inv(image1.header.get_vox2ras_tkr())
    else:
        ras2vox1 = numpy.linalg.inv(image1.header.get_vox2ras())
    
    if tkr:
        vox2ras2 = image2.header.get_vox2ras_tkr()
    else:
        vox2ras2 = image2.header.get_vox2ras()

    xyz1 = brainmesh.coordinates()

    ijk1 = apply_affine(ras2vox1, xyz1)# .T

    if box is not None:
        bounds = get_bounding_box(box)
        dxyz = [bounds[x].start for x in range(3)]
    else:

        dxyz = [0, 0, 0]

    def downscale(points):

        points += npad
        points -= dxyz
        points /= coarsening_factor

        return points

    def upscale(points):

        points *= coarsening_factor
        points -= npad
        points += dxyz

        return points

    assert np.allclose(downscale(upscale(np.array([42, -3.1415, 1e6]))), [42, -3.1415, 1e6])

    transformed_points = np.zeros_like(ijk1) - np.inf
    
    points = downscale(ijk1)    

    print("Iterating over all mesh nodes")

    progress = tqdm(total=points.shape[0])
    for idx in range(points.shape[0]):
        
        try:
            transformed_point = mapping(points[idx, :])
        except RuntimeError:
            print(points[idx, :], "not in BoxMesh")
            exit()
        
        transformed_points[idx, :] = transformed_point

        progress.update(1)
    
    if np.max(transformed_point) > 255:
        raise ValueError

    transformed_points = upscale(transformed_points)

    if np.max(transformed_points) > 255:
        raise ValueError

    transformed_points = np.array(transformed_points)

    xyz2 = apply_affine(vox2ras2, transformed_points)

    brainmesh.coordinates()[:] = xyz2

    return brainmesh



def make_mapping(cubemesh, v, jobfile, hyperparameters, ocd):


    mappings = []

    for coordinate in ["x[0]", "x[1]", "x[2]"]:

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

    vxyz = Function(v.function_space())

    arr = np.zeros_like(vxyz.vector()[:])

    arr[0::3] = mappings[0].vector()[:]
    arr[1::3] = mappings[1].vector()[:]
    arr[2::3] = mappings[2].vector()[:]

    assert np.sum(arr) != 0

    vxyz.vector()[:] = arr
    
    assert norm(vxyz) != 0

    return vxyz