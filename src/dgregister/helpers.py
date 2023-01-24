import pathlib
from fenics import *
from fenics_adjoint import *
import os
# import h5py
import numpy as np
import nibabel
import json


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass


def get_bounding_box_limits(x):
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


def crop_rectangular(imagefiles):

    largest_box = np.zeros((256, 256, 256))


    for mfile in imagefiles:

        # mfile = os.path.join(datapath, pat, "MASKS", "parenchyma.mgz")

        if isinstance(mfile, np.ndarray):
            mask = mfile
        else:
            mask = nibabel.load(mfile).get_fdata()# .astype(bool)

        boundary = get_bounding_box_limits(mask)
        # print(boundary)

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
        assert boundary == get_bounding_box_limits(box)

        largest_box += box

    # bz:
    # why did I comment this out? This should be called to obtain a Rectangular box
    # largest_box = get_bounding_box(largest_box)

    return np.where(largest_box >=1, True, False)


def cut_to_box(image, box_bounds, inverse=False, cropped_image=None):

    # box_boundary = get_bounding_box_limits(box)

    box_boundary = box_bounds

    xlim_box = [box_boundary[0].start, box_boundary[0].stop]
    ylim_box = [box_boundary[1].start, box_boundary[1].stop]
    zlim_box = [box_boundary[2].start, box_boundary[2].stop]
    size = [xlim_box[1] - xlim_box[0], ylim_box[1] - ylim_box[0], zlim_box[1] - zlim_box[0]]
    size = [np.ceil(x).astype(int) for x in size]

    # boundary = get_bounding_box_limits(image)
    # xlim = [boundary[0].start, boundary[0].stop]
    # ylim = [boundary[1].start, boundary[1].stop]
    # zlim = [boundary[2].start, boundary[2].stop]


    # assert size[0] >= xlim[1] - xlim[0]
    # assert size[1] >= ylim[1] - ylim[0]
    # assert size[2] >= zlim[1] -zlim[0]

    # image_center = [xlim[1] + xlim[0], ylim[1] + ylim[0], zlim[1] + zlim[0]]
    # image_center = [int(x / 2) for x in image_center]
    
    
    if inverse:

        raise NotImplementedError

        idx = np.zeros_like(image)
        returnimage = np.zeros_like(image).astype(float)



        idx[image_center[0] - int(size[0] / 2):image_center[0] + int(size[0] / 2),
                    image_center[1] - int(size[1] / 2):image_center[1] + int(size[1] / 2),
                    image_center[2] - int(size[2] / 2):image_center[2] + int(size[2] / 2),
        ] = 1


        returnimage[idx] = cropped_image.flatten()

        return returnimage
    
    else:


        returnimage = np.zeros(tuple(size))

        returnimage = image[xlim_box[0]:xlim_box[1], ylim_box[0]:ylim_box[1], zlim_box[0]:zlim_box[1],]

        # returnimage = image[image_center[0] - int(size[0] / 2):image_center[0] + int(size[0] / 2),
        #             image_center[1] - int(size[1] / 2):image_center[1] + int(size[1] / 2),
        #             image_center[2] - int(size[2] / 2):image_center[2] + int(size[2] / 2),
        # ]

        print("cropped shape", returnimage.shape, "box boundary", box_bounds)

    return returnimage



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


def load_control(hyperparameters, vCG):

    # if not hyperparameters["ocd"]:
    #     assert "Control.hdf" in hyperparameters["starting_guess"]

    assert os.path.isfile(hyperparameters["starting_guess"])

    print_overloaded("Will try to read starting guess")
    # h5file = h5py.File(hyperparameters["starting_guess"])

    readmesh = None

    if hyperparameters["interpolate"]:

        hyperparameterfile = pathlib.Path(hyperparameters["starting_guess"]).parent / "hyperparameters.json"
        
        assert hyperparameterfile.is_file()

        old_params = json.load(open(hyperparameterfile))

        nx = old_params["input.shape"][0]
        ny = old_params["input.shape"][1]
        nz = old_params["input.shape"][2]

        Lx = hyperparameters["input.shape"][0]
        Ly = hyperparameters["input.shape"][1]
        Lz = hyperparameters["input.shape"][2]

        # npad = 
        readmesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

        readmesh.coordinates()[:, 0] *= (Lx / nx)
        readmesh.coordinates()[:, 1] *= (Ly / ny)
        readmesh.coordinates()[:, 2] *= (Lz / nz)

        readVspace = VectorFunctionSpace(readmesh, old_params["velocity_functionspace"], old_params["velocity_functiondegree"])

        for id in range(3):
            for mname, m in zip(["readmesh", "vCG.mesh()"], [readmesh, vCG.mesh()]):
                co = m.coordinates()
                print_overloaded(mname)
            
                print_overloaded("id", id, np.min(co[:, id]), np.max(co[:, id]))

        print_overloaded("nx, ny, nz", nx, ny, nz)
        print_overloaded("Lx, Ly, Lz", Lx, Ly, Lz)


        # xyz = vCG.tabulate_dof_coordinates()
        # utes = Function(vCG)
        # for id in range(xyz.shape[0]):
        #     try:
        #         utes(xyz[id, :])
        #     except RuntimeError:
        #         print_overloaded(id, xyz[id, :], "not in vCG.mesh()")

        # raise NotImplementedError("note so self: remove checks after bug is fixed!")

        coarsev = Function(readVspace)


    else:
        readmesh = vCG.mesh()

        coarsev = Function(vCG)


    hdf = HDF5File(readmesh.mpi_comm(), hyperparameters["starting_guess"], "r")
    hdf.read(coarsev, hyperparameters["readname"])
    hdf.close()

    print_overloaded("max after loading", coarsev.vector()[:].max())
    print_overloaded("Succesfully read starting guess", hyperparameters["starting_guess"])


    if hyperparameters["interpolate"]:

        # coarsev.function_space().mesh() = 

        coarsev_temp = coarsev

        # VTemp = VectorFunctionSpace()

        # coarsev_temp = Function(VTemp)

        # coarsev_temp.vector()[:] 

        print_overloaded("Trying to interpolate")
        finev = interpolate(coarsev_temp, V=vCG)
        print_overloaded("Interpolation worked")    



        return finev

    else:
        return coarsev




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

    vCG = VectorFunctionSpace(domainmesh, hyperparameters["velocity_functionspace"], hyperparameters["velocity_functiondegree"])

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


def store_during_callback(current_iteration, hyperparameters, files, Jd, Jreg, l2loss,
                            domainmesh, velocityField, current_pde_solution, control=None):

    print_overloaded("Iter", format(current_iteration, ".0f"), 
                    "Jd =", format(Jd, ".4e"), 
                    "L2loss =", format(l2loss, ".4e"), "Reg =", format(Jreg, ".2e"))



    if MPI.rank(MPI.comm_world) == 0:
    
        with open(files["lossfile"], "a") as myfile:
            myfile.write(str(float(Jd))+ ", ")
        with open(files["regularizationfile"], "a") as myfile:
            myfile.write(str(float(Jreg))+ ", ")
        with open(files["l2lossfile"], "a") as myfile:
            myfile.write(str(float(l2loss))+ ", ")
    

    # 
    hyperparameters["Jd_current"] = float(Jd)
    hyperparameters["Jreg_current"] = float(Jreg)
    
    
    with XDMFFile(hyperparameters["outputfolder"] + "/State_checkpoint.xdmf") as xdmf:
        xdmf.write_checkpoint(current_pde_solution, "CurrentState", 0.)
    with XDMFFile(hyperparameters["outputfolder"] + "/Velocity_checkpoint.xdmf") as xdmf:
        xdmf.write_checkpoint(velocityField, "CurrentV", 0.)

    file = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/CurrentV.hdf", "w")
    file.write(velocityField, "function")
    file.close()        

    if control is not None:
        file = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/CurrentControl.hdf", "w")
        file.write(control, "function")
        file.close()    

    # DO NOT DELETE 
    # ROUTINE TO STORE TRANSFORMED BRAIN DURING OPTIMIZATION
    # COULD BE USEFUL FOR PLOTTING IN THE FUTURE
    # if min(hyperparameters["input.shape"]) > 1 and len(hyperparameters["input.shape"]) == 3 and (current_iteration % 10 == 0):

    #     retimage = fem2mri(function=current_pde_solution, shape=hyperparameters["input.shape"])
        
    #     if MPI.rank(MPI.comm_world) == 0:
    #         nifti = nibabel.Nifti1Image(retimage, nibabel.load(hyperparameters["input"]).affine)

    #         nibabel.save(nifti, hyperparameters["outputfolder"] + '/mri/state_at_' + str(current_iteration) + '.mgz')

    #         print_overloaded("Stored mgz image of transformed image at iteration", current_iteration)
