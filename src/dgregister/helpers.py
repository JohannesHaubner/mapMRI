from fenics import *
from fenics_adjoint import *
import os
import h5py
import numpy as np
import nibabel

def get_bounding_box(x):
    """ Calculates the bounding box of a ndarray"""
    mask = x == 0
    bbox = []
    all_axis = np.arange(x.ndim)
    for kdim in all_axis:
        nk_dim = np.delete(all_axis, kdim)
        mask_i = mask.all(axis=tuple(nk_dim))
        dmask_i = np.diff(mask_i)
        idx_i = np.nonzero(dmask_i)[0]
        if len(idx_i) != 2:
            raise ValueError(
                'Algorithm failed, {} does not have 2 elements!'.format(idx_i))
        bbox.append(slice(idx_i[0] + 1, idx_i[1] + 1))
    return bbox


def get_largest_box(imagefiles):

    largest_box = np.zeros((256, 256, 256))


    for mfile in imagefiles:

        # mfile = os.path.join(datapath, pat, "MASKS", "parenchyma.mgz")

        mask = nibabel.load(mfile).get_fdata()# .astype(bool)

        boundary = get_bounding_box(mask)
        print(boundary)

        box = np.ones((256, 256, 256))
        
        xlim = [boundary[0].start, boundary[0].stop]
        ylim = [boundary[1].start, boundary[1].stop]
        zlim = [boundary[2].start, boundary[2].stop]

        mx = np.zeros_like(mask)
        mx[xlim[0]:(xlim[1]), ...] = 1

        my = np.zeros_like(mask)
        my[:, ylim[0]:(ylim[1]), :] = 1
        mz = np.zeros_like(mask)
        mz[:, :, zlim[0]:(zlim[1])] = 1

        box *= mx * my * mz
        assert boundary == get_bounding_box(box)

        largest_box += box

    # largest_box = get_bounding_box(largest_box)

    return largest_box





def cut_to_box(image, box):

    box_boundary = get_bounding_box(box)
    xlim_box = [box_boundary[0].start, box_boundary[0].stop]
    ylim_box = [box_boundary[1].start, box_boundary[1].stop]
    zlim_box = [box_boundary[2].start, box_boundary[2].stop]
    
    size = [xlim_box[1] - xlim_box[0], ylim_box[1] - ylim_box[0], zlim_box[1] - zlim_box[0]]
    size = [np.ceil(x).astype(int) for x in size]


    cropped_image = np.zeros(tuple(size))

    boundary = get_bounding_box(image)

    xlim = [boundary[0].start, boundary[0].stop]
    ylim = [boundary[1].start, boundary[1].stop]
    zlim = [boundary[2].start, boundary[2].stop]

    assert size[0] >= xlim[1] - xlim[0]
    assert size[1] >= ylim[1] - ylim[0]
    assert size[2] >= zlim[1] -zlim[0]

    image_center = [xlim[1] + xlim[0], ylim[1] + ylim[0], zlim[1] + zlim[0]]
    image_center = [int(x / 2) for x in image_center]

    cropped_image = image[image_center[0] - int(size[0] / 2):image_center[0] + int(size[0] / 2),
                image_center[1] - int(size[1] / 2):image_center[1] + int(size[1] / 2),
                image_center[2] - int(size[2] / 2):image_center[2] + int(size[2] / 2),
    ]

    print("cropped shape", cropped_image.shape)

    return cropped_image



def pad_with(vector, pad_width, iloc, kwargs):

    pad_value = kwargs.get('padder', 0)

    vector[:pad_width[0]] = pad_value

    vector[-pad_width[1]:] = pad_value

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

def get_lumped_mass_matrices(vCG):

    s1 = TrialFunction(vCG)
    s2 = TestFunction(vCG)
    form = inner(s1, s2) * dx
    mass_action_form = action(form, Constant((1., ) * Function(vCG).geometric_dimension()))
    M_lumped = assemble(form)
    M_lumped_inv = assemble(form)
    M_lumped.zero()
    M_lumped_inv.zero()
    diag = assemble(mass_action_form)
    diag[:] = np.sqrt(diag[:])
    diaginv = assemble(mass_action_form)
    diaginv[:] = 1.0 / diag[:]
    M_lumped.set_diagonal(diag)
    M_lumped_inv.set_diagonal(diaginv)

    return M_lumped, M_lumped_inv


def load_velocity(hyperparameters, controlfun):

    assert os.path.isfile(hyperparameters["starting_guess"])

    print_overloaded("Will try to read starting guess")
    h5file = h5py.File(hyperparameters["starting_guess"])

    print_overloaded("keys in h5file", list(h5file.keys()))

    if controlfun is not None:
        print_overloaded("max before loading", controlfun.vector()[:].max())
        working_mesh = controlfun.function_space().mesh()
        print_overloaded("trying to read velocity without loading mesh")
        hdf = HDF5File(working_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')

        
    else:
        print_overloaded("Reading mesh")
        working_mesh = Mesh()
        hdf = HDF5File(working_mesh.mpi_comm(), hyperparameters["starting_guess"], 'r')
        hdf.read(working_mesh, "/mesh", False)
        
        vCG = VectorFunctionSpace(working_mesh, hyperparameters["velocity_functionspace"], hyperparameters["functiondegree"])
        controlfun = Function(vCG)

    print_overloaded("trying to read", hyperparameters["readname"])
    hdf.read(controlfun, hyperparameters["readname"])
    hdf.close()

    print_overloaded("max after loading", controlfun.vector()[:].max())
    print_overloaded("Succesfully read starting guess")

    return working_mesh, vCG, controlfun





def interpolate_velocity(hyperparameters, domainmesh, vCG, controlfun, store_pvd=False):

    # controlFile = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Loaded_Control.hdf", "w")
    # controlFile.write(domainmesh, "mesh")
    # controlFile.write(controlfun, "loaded_control")
    # controlFile.close()

    # with XDMFFile(hyperparameters["outputfolder"] + "/Loaded_Control.xdmf") as xdmf:
    #     xdmf.write_checkpoint(controlfun, "coarse", 0.)
    if store_pvd:
        File(hyperparameters["outputfolder"] + "/Loaded_Control.pvd") << controlfun

    print_overloaded("parameters['ghost_mode'] in interpolate_velocity()", parameters['ghost_mode'])
    print_overloaded("trying to refine the mesh")

    domainmesh = refine(domainmesh, redistribute=False)

    vCG = VectorFunctionSpace(domainmesh, hyperparameters["velocity_functionspace"], hyperparameters["functiondegree"])

    controlfun_fine = interpolate(controlfun, vCG)

    # l2 = assemble((controlfun_fine - controlfun) ** 2 * dx(domain=controlfun.function_space().mesh()) )
    # l2norm = assemble((controlfun) ** 2 * dx(domain=controlfun.function_space().mesh()) )

    # print_overloaded("COARSE MESH")
    # print_overloaded("L2 error", l2)
    # print_overloaded("rel L2 error", l2 / l2norm)
    # print_overloaded("L2 norm of control", l2norm)

    print_overloaded("FINE MESH")
    
    l2 = assemble((controlfun_fine - controlfun) ** 2 * dx(domain=controlfun_fine.function_space().mesh()) )
    l2norm = assemble((controlfun) ** 2 * dx(domain=controlfun_fine.function_space().mesh()) )

    print_overloaded("L2 error", l2)
    print_overloaded("rel L2 error", l2 / l2norm)
    print_overloaded("L2 norm of control", l2norm)


    print_overloaded("Interpolated expression, writing...")

    # with XDMFFile(hyperparameters["outputfolder"] + "/Interpolated_Control.xdmf") as xdmf:
    #     xdmf.write_checkpoint(controlfun_fine, "fine", 0.)


    if store_pvd:
        File(hyperparameters["outputfolder"] + "/Interpolated_Control.pvd") << controlfun_fine

    controlFileInterpolated = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/Interpolated_Control.hdf", "w")
    controlFileInterpolated.write(domainmesh, "mesh")
    controlFileInterpolated.write(controlfun_fine, "refined_control")
    controlFileInterpolated.close()

    if hyperparameters["debug"]:
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--debug is set, exiting here")
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--------------------------------------------------------------------------------------------------------")
        print_overloaded("--------------------------------------------------------------------------------------------------------")

        exit()

    return domainmesh, vCG, controlfun_fine