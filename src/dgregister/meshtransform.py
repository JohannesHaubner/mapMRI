import json
import os
import pathlib

if "home/bastian" not in os.getcwd():
    from tqdm import tqdm

import nibabel
import numpy
import numpy as np
from fenics import *

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass

from dolfin_adjoint import *
from nibabel.affines import apply_affine
from dgregister.helpers import get_bounding_box_limits, cut_to_box # get_largest_box, pad_with, cut_to_box, 



def map_mesh(xmlfile1: str, imgfile1: str, imgfile2: str, mapping: Function,  
    coarsening_factor: int, npad: int, box: np.ndarray=None, 
    inverse_affine:bool = False, registration_affine: np.ndarray = None,
    outfolder=None, raise_errors: bool = True):

    assert norm(mapping) != 0

    comm = MPI.comm_world
    nprocs = comm.Get_size()

    # MPI.comm_world.Get_size()

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
        print_overloaded("Using trk RAS")
        ras2vox1 = numpy.linalg.inv(image1.header.get_vox2ras_tkr())
    else:
        ras2vox1 = numpy.linalg.inv(image1.header.get_vox2ras())
    
    if tkr:
        vox2ras2 = image2.header.get_vox2ras_tkr()
    else:
        vox2ras2 = image2.header.get_vox2ras()

    xyz1 = brainmesh.coordinates()

    ijk1 = apply_affine(ras2vox1, xyz1)# .T

    
    if registration_affine is not None:
        aff = registration_affine


        if inverse_affine:
            print_overloaded("Using inverse of affine for testing")
            aff = np.linalg.inv(aff)

        ijk1 = apply_affine(aff, ijk1)

    # breakpoint()

    if box is not None:
        bounds = get_bounding_box_limits(box)
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
    
    ijk1 = downscale(ijk1)    

    print_overloaded("Iterating over all mesh nodes")

    if "home/bastian" not in os.getcwd():
        progress = tqdm(total=ijk1.shape[0])
    
    for idx in range(ijk1.shape[0]):
        
        try:
            transformed_point = mapping(ijk1[idx, :])
        except RuntimeError:
            print_overloaded(ijk1[idx, :], "not in BoxMesh")
            # breakpoint()
            raise ValueError("Some ijk1[idx, :] is not in BoxMesh, probably something is wrong.")
            # exit()
        
        transformed_points[idx, :] = transformed_point
        if "home/bastian" not in os.getcwd():
            progress.update(1)

    def in_mri(x):
        return (np.max(x) < 256) and (np.min(x) >= 0)
    
    if (not in_mri(transformed_points)) and raise_errors:
        raise ValueError


    if (not in_mri(transformed_points)) and (not raise_errors):
        print_overloaded("Before upscaling")
        idx = np.sum(transformed_points>255, axis=1).astype(bool)
        print_overloaded(transformed_points[idx, :])
        print_overloaded(idx.sum(), "/", idx.size, "points are outside of [0, 256]")

    transformed_points = upscale(transformed_points)

    distance = np.linalg.norm(ijk1-transformed_points, ord=2, axis=-1)
    
    print_overloaded("Minimum distance travelled ", np.min(distance))
    print_overloaded("Maximum distance travelled ", np.max(distance))

    print_overloaded("Compare to mesh size:", brainmesh.hmin(), brainmesh.hmax(), )

    if (not in_mri(transformed_points)) and raise_errors:
        raise ValueError

    if (not in_mri(transformed_points)) and (not raise_errors):
        print_overloaded("After upscaling")
        idx = np.sum(transformed_points>255, axis=1).astype(bool)

        print_overloaded(idx.sum(), "/", idx.size, "points are outside of [0, 256]")
        print_overloaded(transformed_points[idx, :])

    # breakpoint()
    # transformed_points = np.array(transformed_points)

    xyz2 = apply_affine(vox2ras2, transformed_points)

    brainmesh.coordinates()[:] = xyz2

    return brainmesh



def make_mapping(cubemesh, velocities, state_space, state_degree, parserargs, ocd, dgtransport: bool = False):

    mappings = []

    for coordinate in ["x[0]", "x[1]", "x[2]"]:

        print_overloaded("Transporting, ", coordinate, "coordinate")

        if ocd:
            from dgregister import find_velocity
            velocity = velocities[0]
            V1 = FunctionSpace(cubemesh, "CG", 1)
            
            unity2 = Function(V1) #¤ cubeimg.function_space())
            unity2.vector()[:] = 0

            xin = interpolate(Expression(coordinate, degree=1), V1) # cubeimg.function_space())
            print_overloaded("Running OCD forward pass")
            xout, _, _ = find_velocity.find_velocity(Img=xin, Img_goal=unity2, hyperparameters=hyperparameters, files=[], phi_eval=-velocity, projection=False)

        else:

            # import dgregister.config as config

            # config.hyperparameters = {**hyperparameters, **config.hyperparameters}
            
            cgtransport = not dgtransport

            if cgtransport:
                from dgregister.DGTransport import CGTransport as Transport
                print("Using CG transport")
                V1 = FunctionSpace(cubemesh, "CG", 1)
            else:
                from dgregister.DGTransport import DGTransport as Transport

                print("Using DG transport")
                V1 = FunctionSpace(cubemesh, state_space, state_degree)
            
            xin = interpolate(Expression(coordinate, degree=1), V1) # cubeimg.function_space())
            
            print_overloaded("Interpolated ", coordinate)


            idx = 0
            for key, (velocity, hyperparameters) in velocities.items():

                if hyperparameters["smoothen"]:

                    assert "VelocityField" in parserargs["velocityfilename"] or "CurrentV" in parserargs["velocityfilename"]
                    assert "Control.hdf" not in parserargs["velocityfilename"]


                assert norm(velocity) > 0

                print_overloaded("Calling Transport()", "cgtransport=", cgtransport, "velocity field", idx + 1 / len(velocities))

                if MPI.rank(MPI.comm_world) == 0:
                    idx += 1

                xout = Transport(Img=xin, Wind=-velocity, preconditioner=hyperparameters["preconditioner"], 
                                MaxIter=hyperparameters["max_timesteps"], DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                                solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"])

                xin = xout

            if not cgtransport:
                V1_CG = FunctionSpace(cubemesh, "CG", 1)
                xout = project(xout, V1_CG)
            
                print_overloaded("Projected xout to CG1")


        assert norm(xout) != 0

        mappings.append(xout)

    assert len(mappings) == 3

    assert True not in [norm(x) == 0 for x in mappings]

    vxyz = Function(velocity.function_space())

    arr = np.zeros_like(vxyz.vector()[:])

    arr[0::3] = mappings[0].vector()[:]
    arr[1::3] = mappings[1].vector()[:]
    arr[2::3] = mappings[2].vector()[:]

    assert np.sum(arr) != 0

    vxyz.vector()[:] = arr
    
    assert norm(vxyz) != 0

    return vxyz