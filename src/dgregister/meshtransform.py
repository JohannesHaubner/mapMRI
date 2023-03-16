import json
import os
import pathlib

from tqdm import tqdm
import numpy.linalg
import nibabel
import numpy
import numpy as np
from fenics import *
try:
    import pymesh
except:
    print("Could not import pymesh")


try:
    import meshio
except:
    print("Could not import meshio")

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
from dgregister.DGTransport import DGTransport
import dgregister
import dgregister.helpers


def fix_mesh(mesh, detail="normal"):
# This function is from
# https://github.com/PyMesh/PyMesh/blob/main/scripts/fix_mesh.py
    bbox_min, bbox_max = mesh.bbox
    diag_len = numpy.linalg.norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2
    print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 10: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh



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


def map_mesh(mappings: list, noaffine: bool, 
            data: dgregister.helpers.Data, update:bool = False, 
            remesh: bool=False, tmpdir=None,
            raise_errors=True):

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
    if len(mappings) > 0:
        assert np.max(input_mesh_xyz_vox) <= np.max(mappings[0].function_space().mesh().coordinates())
    # breakpoint()

    print("Downscaled brain mesh vox coordinates to cube mesh coordinate system")

    # TODO FIXME why was this 4 ? Does it make sense?
    if data.pad not in [0, 2]:
        # TODO FIXME why was this 4 ? Does it make sense?
        raise ValueError

    
    # NOTE
    # before the registration affine was applied first.

    target_mesh_xyz_vox = np.copy(input_mesh_xyz_vox)
    reuse_mesh = True

    for mapping_idx, mapping in enumerate(mappings):

        assert norm(mapping) != 0

        deformed_mesh_ijk = np.zeros_like(target_mesh_xyz_vox)

        print_overloaded("Iterating over all mesh nodes, mapping", mapping_idx + 1, "/", len(mappings))

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


        if remesh:
            if reuse_mesh:
                tempmesh = data.meshcopy()
            else:
                tempmesh = Mesh(fixed_surfacemesh_file.replace(".stl", ".xml"))

            tempmesh.coordinates()[:] = deformed_mesh_ijk

            if "boundary" not in data.input_meshfile:
                surfacemesh = BoundaryMesh(tempmesh, "exterior")
            else:
                surfacemesh = tempmesh

            tmpfile = tmpdir + "boundary_trafo" + str(mapping_idx) + ".xml"
            # Convert xml to stl with meshio:
            File(tmpfile) << surfacemesh
            assert os.path.isfile(tmpfile)

            meshio_mesh = meshio.read(tmpfile)
            meshio_mesh.write(tmpfile.replace(".xml", ".stl"))

            print("Stored boundary mesh to stl with meshio")

            # Load, fix and store using pymesh:
            pymesh_mesh = pymesh.meshio.load_mesh(tmpfile.replace(".xml", ".stl"))
            print("Loaded mesh to pymesh")
            fixed_pymesh_surface_mesh = fix_mesh(pymesh_mesh, detail="normal")
            print("Fixed surface mesh with pymesh")
            fixed_surfacemesh_file = tmpfile.replace(".xml", "_fixed.stl")
            pymesh.meshio.save_mesh(fixed_surfacemesh_file, fixed_pymesh_surface_mesh)
            print("wrote fixed mesh with pymesh")

            # Load stl and convert back to xml with  meshio:
            fixed_meshio_mesh = meshio.read(fixed_surfacemesh_file)
            fixed_meshio_mesh.write(fixed_surfacemesh_file.replace(".stl", ".xml"))
            print("Converted fixed mesh back to xml")
            # Load back to fenics:
            fixed_surface_mesh = Mesh(fixed_surfacemesh_file.replace(".stl", ".xml"))
            print("Loaded fixed mesh back to fenics")

            print(fixed_surface_mesh.coordinates()[:].shape, deformed_mesh_ijk.shape)
            
            if not fixed_surface_mesh.coordinates()[:].shape == input_mesh_xyz_vox.shape:
                reuse_mesh = False

            deformed_mesh_ijk = fixed_surface_mesh.coordinates()[:]

            ###
            outcoords = upscale(np.copy(deformed_mesh_ijk), npad=data.pad, dxyz=data.dxyz)
            print("Upscaled from cube mesh coordinate system to image voxel coordinates")

            outcoords = apply_affine(data.registration_affine, outcoords)
            outcoords = apply_affine(data.vox2ras_target,  outcoords)
            # outcoords = apply_affine(data.vox2ras_input,  outcoords)
            
            mesh_for_visual = Mesh(fixed_surfacemesh_file.replace(".stl", ".xml"))
            mesh_for_visual.coordinates()[:] = outcoords

            vizmeshfile = fixed_surfacemesh_file.replace(".stl", "_correctcoords.xml")
            File(vizmeshfile) << mesh_for_visual
            transormed_xmlmesh = meshio.read(vizmeshfile)
            transormed_xmlmesh.write(vizmeshfile.replace(".xml", ".xdmf"))
            
            ####
        
        target_mesh_xyz_vox = deformed_mesh_ijk
        
    # Now scale back up
    # breakpoint()

    target_mesh_xyz_vox = upscale(target_mesh_xyz_vox, npad=data.pad, dxyz=data.dxyz)
    print("Upscaled from cube mesh coordinate system to image voxel coordinates")


    target_mesh_xyz_vox1 =  np.copy(target_mesh_xyz_vox)

    if not noaffine:
        print("Applying registration affine")
        target_mesh_xyz_vox1 = apply_affine(data.registration_affine, np.copy(target_mesh_xyz_vox1))
    else:
        print("*"*80)
        print("Not applying registration affine")
        print("*"*80)

    target_mesh_xyz_ras1 = apply_affine(data.vox2ras_target,  target_mesh_xyz_vox1)
    
    if reuse_mesh:
        targetmesh1 = data.meshcopy()
    else:
        targetmesh1 = fixed_surface_mesh
    
    targetmesh1.coordinates()[:] = target_mesh_xyz_ras1

    # target_mesh_xyz_vox2 =  np.copy(target_mesh_xyz_vox)
    # target_mesh_xyz_vox2 = apply_affine(np.linalg.inv(data.registration_affine), np.copy(target_mesh_xyz_vox2))
    # target_mesh_xyz_ras2 = apply_affine(data.vox2ras_target, target_mesh_xyz_vox2)
    # targetmesh2 = data.meshcopy()
    # targetmesh2.coordinates()[:] = target_mesh_xyz_ras2


    return targetmesh1 # , targetmesh2



def make_mapping(cubemesh, control, M_lumped_inv, hyperparameters,):

    import dgregister.preconditioning

    if str(hyperparameters["slurmid"]) == "450276":
        dgregister.preconditioning.omega = 0.5
        dgregister.preconditioning.epsilon = 0.5
    elif str(hyperparameters["slurmid"]) not in ["446152", "446600", "447918"]:
        raise NotImplementedError("Check which omega, epsilon was used.")

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

        print_overloaded("Using CGTransport")
        xout = CGTransport(Img=xin, Wind=-velocity, 
                           DeltaT=hyperparameters["DeltaT"], 
                           preconditioner="amg", 
                        MaxIter=hyperparameters["max_timesteps"], timestepping=hyperparameters["timestepping"], 
                        solver="krylov", MassConservation=hyperparameters["MassConservation"])
        
        # print_overloaded("Using DGTransport")
        # xout = DGTransport(Img=xin, Wind=-velocity, 
        #                    DeltaT=hyperparameters["DeltaT"], 
        #                    preconditioner="amg", 
        #                 MaxIter=hyperparameters["max_timesteps"], timestepping=hyperparameters["timestepping"], 
        #                 solver="krylov", MassConservation=hyperparameters["MassConservation"])
        
        
        
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