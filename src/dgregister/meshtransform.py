import json
import os
import pathlib
import numpy
import numpy as np
from fenics import *
from fenics_adjoint import *

import SVMTK as svmtk
import meshio

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass


from nibabel.affines import apply_affine
from dgregister.preconditioning_overloaded import preconditioning
from dgregister.transformation_overloaded import transformation
from dgregister.CGTransport import CGTransport
from dgregister.helpers import get_lumped_mass_matrices




def downscale(points: np.ndarray, npad: np.ndarray, dxyz: np.ndarray) -> np.ndarray:

    points += npad
    points -= dxyz

    return points

def upscale(points: np.ndarray, npad: np.ndarray, dxyz: np.ndarray) -> np.ndarray:

    points -= npad
    points += dxyz

    return points



def map_mesh(mappings: list, data, 
            remesh: bool=False, exportdir=None,):

    comm = MPI.comm_world
    nprocs = comm.Get_size()

    if nprocs != 1:
        raise NotImplementedError("This function needs to be run sequentially")

    input_mesh_xyz_ras = data.meshcopy().coordinates()

    input_mesh_xyz_vox = apply_affine(np.linalg.inv(data.vox2ras_input), input_mesh_xyz_ras)

    assert np.min(input_mesh_xyz_vox) >= 0
    assert np.max(input_mesh_xyz_vox) <= 256

    input_mesh_xyz_vox = downscale(input_mesh_xyz_vox, npad=data.pad, dxyz=data.dxyz)

    if remesh:

        tempmesh = data.meshcopy()
        tempmesh.coordinates()[:] = input_mesh_xyz_vox

        tmpfile = exportdir + "inputmesh_downscaled.xml"

        File(tmpfile) << tempmesh

        tempmesh = meshio.read(tmpfile)
        tempmesh.write(tmpfile.replace(".xml", ".stl"))
        tempmesh.write(tmpfile.replace(".xml", ".xdmf"))

        del tempmesh, tmpfile

    assert np.min(input_mesh_xyz_vox) >= 0
    if len(mappings) > 0:
        assert np.max(input_mesh_xyz_vox) <= np.max(mappings[0].function_space().mesh().coordinates())

    print("Downscaled brain mesh vox coordinates to cube mesh coordinate system")

    if data.pad not in [0, 2]:
        raise ValueError

    
    target_mesh_xyz_vox = np.copy(input_mesh_xyz_vox)
    reuse_mesh = True

    for mapping_idx, mapping in enumerate(mappings):

        assert norm(mapping) != 0

        deformed_mesh_ijk = np.zeros_like(target_mesh_xyz_vox)

        print_overloaded("Iterating over all mesh nodes, mapping", mapping_idx + 1, "/", len(mappings))

        print_at = [int(target_mesh_xyz_vox.shape[0] * (x + 1) / 10) for x in range(10)]

        for idx in range(target_mesh_xyz_vox.shape[0]):
            
            if idx in print_at:
                print("----", int(100 * idx / target_mesh_xyz_vox.shape[0]), "%")

            try:
                transformed_point = mapping(target_mesh_xyz_vox[idx, :])
            except RuntimeError:
                print_overloaded(target_mesh_xyz_vox[idx, :], "not in BoxMesh")
                raise ValueError("Some ijk1[idx, :] is not in BoxMesh, probably something is wrong.")
            
            deformed_mesh_ijk[idx, :] = transformed_point
            

        if remesh:

            if reuse_mesh:
                tempmesh = data.meshcopy()
            else:
                tempmesh = Mesh(fixed_surfacemesh_file.replace(".stl", ".xml"))

            tempmesh.coordinates()[:] = deformed_mesh_ijk
            print("Assigned new coordinates to mesh")

            if tempmesh.topology().dim() != 2:
                print("Created Boundarymesh from tempesh tempmesh")
                surfacemesh = BoundaryMesh(tempmesh, "exterior")
            else:
                surfacemesh = tempmesh

            tmpfile = exportdir + "boundary_trafo" + str(mapping_idx) + ".xml"
            # Convert xml to stl with meshio:
            File(tmpfile) << surfacemesh
            print("Stored current mesh to xml")
            assert os.path.isfile(tmpfile)

            stlfile = tmpfile.replace(".xml", ".stl")


            meshio_mesh = meshio.read(tmpfile)
            meshio_mesh.write(stlfile)


            print("Stored boundary mesh to stl with meshio")

            fixed_surfacemesh_file = stlfile.replace(".stl", "_fixed.stl")

            surface = svmtk.Surface(stlfile)

            print("Loaded surface with SVMTK")

            # Remesh surface
            surface.isotropic_remeshing(1, 3, False)

            print("Performed isotropic remeshing")

            surface.smooth_taubin(2)

            print("Performed taubin smoothing")

            surface.fill_holes()
            print("Filled holes")

            # Separate narrow gaps
            # Default argument is -0.33. 
            surface.separate_narrow_gaps(-0.33)
            print("separated gaps")

            surface.save(fixed_surfacemesh_file)

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
            
            mesh_for_visual = Mesh(fixed_surfacemesh_file.replace(".stl", ".xml"))
            mesh_for_visual.coordinates()[:] = outcoords

            vizmeshfile = fixed_surfacemesh_file.replace(".stl", "_correctcoords.xml")
            File(vizmeshfile) << mesh_for_visual
            transormed_xmlmesh = meshio.read(vizmeshfile)
            transormed_xmlmesh.write(vizmeshfile.replace(".xml", ".xdmf"))
            transormed_xmlmesh.write(vizmeshfile.replace(".xml", ".stl"))
            
            ####
        
        target_mesh_xyz_vox = deformed_mesh_ijk
        
    # Now scale back up


    target_mesh_xyz_vox = upscale(target_mesh_xyz_vox, npad=data.pad, dxyz=data.dxyz)
    print("Upscaled from cube mesh coordinate system to image voxel coordinates")

    target_mesh_xyz_vox1 =  np.copy(target_mesh_xyz_vox)

    target_mesh_xyz_ras1 = apply_affine(data.vox2ras_target,  target_mesh_xyz_vox1)
    
    if reuse_mesh:
        targetmesh1 = data.meshcopy()
    else:
        targetmesh1 = fixed_surface_mesh
    
    targetmesh1.coordinates()[:] = target_mesh_xyz_ras1

    return targetmesh1



def make_mapping(cubemesh, control, hyperparameters):
    """

    Transports the field (x, y, z) defined on a cube mesh by solving the (vector-) transport equation with
    the vector field (x, y, z) as initial condition.

    Args:
        cubemesh (dolfin.mesh): mesh
        control (dolfin.VectorFunction): Control variable
        hyperparameters (dict): numerical hyperparameters

    Returns:
        dolfin.VectorFunction: The transported coordinate field (X, Y, Z)
    """


    _, M_lumped_inv = get_lumped_mass_matrices(vCG=control.function_space())

    mappings = []

    control_L2 = transformation(control, M_lumped_inv)
    print_overloaded("Preconditioning L2_controlfun, name=", control_L2)
    velocity = preconditioning(control_L2)

    # Sanity check
    assert norm(velocity) > 0

    V1 = FunctionSpace(cubemesh, "CG", 1)

    for coordinate in ["x[0]", "x[1]", "x[2]"]:

        print_overloaded("Transporting, ", coordinate, "coordinate")     

        xin = interpolate(Expression(coordinate, degree=1), V1)
        
        # "Using CGTransport to transport coordinates. This is useful since there are no jumps in the field (x, y, z)."

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