import json
import os
import pathlib

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

from fenics_adjoint import *
from nibabel.affines import apply_affine
from dgregister.helpers import get_bounding_box_limits, cut_to_box # get_largest_box, pad_with, cut_to_box, 
from dgregister.preconditioning_overloaded import preconditioning
from dgregister.transformation_overloaded import transformation
from dgregister.CGTransport import CGTransport
import dgregister
import dgregister.helpers

def downscale(points: np.ndarray, npad: np.ndarray, dxyz: np.ndarray) -> np.ndarray:

    points += npad
    points -= dxyz

    return points

def upscale(points: np.ndarray, npad: np.ndarray, dxyz: np.ndarray) -> np.ndarray:

    points -= npad
    points += dxyz

    return points




# def affine_transformation(affine: np.ndarray, xyz: np.ndarray,) -> np.ndarray:
#     """
#     """

#     comm = MPI.comm_world
#     nprocs = comm.Get_size()

#     if nprocs != 1:
#         raise NotImplementedError
    
#     mesh_xyz = np.copy(xyz)

#     mesh_xyz = apply_affine(affine, mesh_xyz)

#     return mesh_xyz


def map_mesh(mappings: list, 
            data: dgregister.helpers.Data, update:bool = False, 
            raise_errors=True):

    if not len(mappings) == 1:
        # TODO FIXME
        # TODO check hat the order of the mappings is reversed.
        raise NotImplementedError    

    comm = MPI.comm_world
    nprocs = comm.Get_size()

    if nprocs != 1:
        raise NotImplementedError
    

    input_mesh_xyz_ras = data.meshcopy().coordinates()

    input_mesh_xyz_vox = apply_affine(np.linalg.inv(data.vox2ras_input), input_mesh_xyz_ras)

    assert np.min(input_mesh_xyz_vox) >= 0
    assert np.max(input_mesh_xyz_vox) <= 256
    # breakpoint()

    input_mesh_xyz_vox = downscale(input_mesh_xyz_vox, npad=data.pad, dxyz=data.dxyz)

    assert np.min(input_mesh_xyz_vox) >= 0
    assert np.max(input_mesh_xyz_vox) <= np.max(mappings[0].function_space().mesh().coordinates())
    # breakpoint()

    print("Downscaled brain mesh vox coordinates to cube mesh coordinate system")

    # TODO FIXME why was this 4 ? Does it make sense?
    if data.pad not in [0, 2]:
        # TODO FIXME why was this 4 ? Does it make sense?
        raise ValueError

    
    # NOTE
    # before the registration affine was applied first.

    target_mesh_xyz_vox = input_mesh_xyz_vox

    for idx, mapping in enumerate(mappings):

        # print("Testing, continue")
        # continue

        assert norm(mapping) != 0

        deformed_mesh_ijk = np.zeros_like(target_mesh_xyz_vox)

        print_overloaded("Iterating over all mesh nodes, mapping", idx + 1, "/", len(mappings))

        if update:
         progress = tqdm(total=int(target_mesh_xyz_vox.shape[0]))

        for idx in range(target_mesh_xyz_vox.shape[0]):
            
            try:
                transformed_point = mapping(target_mesh_xyz_vox[idx, :])
            except RuntimeError:
                print_overloaded(target_mesh_xyz_vox[idx, :], "not in BoxMesh")
                raise ValueError("Some ijk1[idx, :] is not in BoxMesh, probably something is wrong.")
            
            deformed_mesh_ijk[idx, :] = transformed_point
            
            if update:
                progress.update(1)

        target_mesh_xyz_vox = deformed_mesh_ijk
        # TODO FIXME
        # coud save the mesh here and do remeshing.

    # Now scale back up
    # breakpoint()

    target_mesh_xyz_vox = upscale(target_mesh_xyz_vox, npad=data.pad, dxyz=data.dxyz)
    print("Upscaled from cube mesh coordinate system to image voxel coordinates")

    # Inverse of registration.
    # TODO
    # FIXME
    # TODO
    # check if inverse needs to be applied here.
    target_mesh_xyz_vox1 =  np.copy(target_mesh_xyz_vox)
    target_mesh_xyz_vox1 = apply_affine(data.registration_affine, np.copy(target_mesh_xyz_vox1))
    target_mesh_xyz_ras1 = apply_affine(data.vox2ras_target,  target_mesh_xyz_vox1)
    targetmesh1 = data.meshcopy()
    targetmesh1.coordinates()[:] = target_mesh_xyz_ras1

    target_mesh_xyz_vox2 =  np.copy(target_mesh_xyz_vox)
    target_mesh_xyz_vox2 = apply_affine(np.linalg.inv(data.registration_affine), np.copy(target_mesh_xyz_vox2))
    target_mesh_xyz_ras2 = apply_affine(data.vox2ras_target, target_mesh_xyz_vox2)
    targetmesh2 = data.meshcopy()
    targetmesh2.coordinates()[:] = target_mesh_xyz_ras2


    return targetmesh1, targetmesh2



def make_mapping(cubemesh, control, M_lumped_inv, hyperparameters,):

    mappings = []

    control_L2 = transformation(control, M_lumped_inv)
    print_overloaded("Preconditioning L2_controlfun, name=", control_L2)
    velocity = preconditioning(control_L2)

    V1 = FunctionSpace(cubemesh, "CG", 1)

    for coordinate in ["x[0]", "x[1]", "x[2]"]:

        print_overloaded("Transporting, ", coordinate, "coordinate")     

        xin = interpolate(Expression(coordinate, degree=1), V1) # cubeimg.function_space())
        
        print_overloaded("Interpolated ", coordinate)

        assert norm(velocity) > 0

        xout = CGTransport(Img=xin, Wind=-velocity, 
                           DeltaT=hyperparameters["DeltaT"], 
                           preconditioner="amg", 
                        MaxIter=hyperparameters["max_timesteps"], timestepping=hyperparameters["timestepping"], 
                        solver="krylov", MassConservation=hyperparameters["MassConservation"])

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