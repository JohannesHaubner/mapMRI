from fenics import *
from fenics_adjoint import *
import os
import numpy
import numpy as np
import nibabel
import pathlib
from parse import parse
import matplotlib.pyplot as plt

from nibabel.affines import apply_affine




def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass


def move_mesh(meshfile, ltafile, inverse, vox2ras, vox2ras2, allow_ras2ras):
    
    regaff = read_vox2vox_from_lta(ltafile, allow_ras2ras)
    fenicsmesh = Mesh(meshfile)
    
    if inverse:
        regaff = numpy.linalg.inv(regaff)
    
    regaff = np.matmul((vox2ras2), np.matmul(regaff, np.linalg.inv(vox2ras)))

    xyz = np.copy(fenicsmesh.coordinates()[:])
    fenicsmesh.coordinates()[:] = apply_affine(regaff, xyz)

    return fenicsmesh



def crop_to_original(orig_image: numpy.ndarray, cropped_image: numpy.ndarray, box: numpy.ndarray, space: int, pad: int):

    box_bounds = get_bounding_box_limits(box)

    limits2 = []
    for l in box_bounds:
        limits2.append(slice(l.start -space, l.stop + space, None))

    box_bounds = limits2

    fillarray = np.zeros_like(orig_image)

    filled_image = cut_to_box(image=fillarray, box_bounds=box_bounds, inverse=True, cropped_image=cropped_image, pad=pad)

    return filled_image


def store2mgz(imgfile1, imgfile2, ijk1, outfolder):
    assert os.path.isdir(outfolder)

    data = np.zeros((256, 256, 256))

    for idx, img in enumerate([imgfile1, imgfile2]):
        image1 = nibabel.load(img)
                 
        i1, j1, k1 = np.rint(ijk1).astype("int")

        data[i1, j1, k1] = 1.

        assert data.sum() > 100

        nii = nibabel.Nifti1Image(data, image1.affine)
        nibabel.save(nii, outfolder + pathlib.Path(imgfile1).name.replace(".mgz", "affine" + str(idx) + "_meshindices.mgz"))



def view(images, axis=1, idx=100, colorbar=False):

    line = "freeview "

    for img in images:

        

        plt.figure()

        plt.title(img)        

        if not isinstance(img, np.ndarray):
            line += img + " "
            print(img)
            img = nibabel.load(img)
            print(img.affine)
            print()
            
            img = img.get_fdata()

        if not (np.allclose(img.shape[axis], 256)):
            raise NotImplementedError
            idx2 = int(img.shape[axis] * idx / 255)
            print(idx, "-->", idx2)
        else:
            idx2 = idx

        plt.imshow(np.take(img, idx2, axis), cmap="Greys_r", vmax=100)
        # 
    print(line)
    plt.show()


def read_vox2vox_from_lta(lta, allow_ras2ras=False):
    File = open(lta)

    lines = File.readlines()

    regmatrix_v2v = []

    v2v = False
    print("Reading matrix from", lta)
    print("*"*80)
    for line in lines:
        # print(line)

        if "LINEAR_VOX_TO_VOX" in line:
            v2v = True

        res = parse("{} {} {} {}", line.replace("\n", ""))

        try:
            a, b, c, d = float(res[0]), float(res[1]), float(res[2]), float(res[3])
            print(a,b,c,d)
            

            regmatrix_v2v.append([a,b,c,d])
        except:
            pass
    
    print("*"*80)
    if not allow_ras2ras:
        assert v2v

    File.close()

    # embed()


    regmatrix_v2v = np.array(regmatrix_v2v)

    return regmatrix_v2v


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


def get_larget_box(imagefiles):

    largest_box = np.zeros((256, 256, 256))


    for mfile in imagefiles:

        if isinstance(mfile, np.ndarray):
            mask = mfile
        else:
            mask = nibabel.load(mfile).get_fdata()# .astype(bool)

        boundary = get_bounding_box_limits(mask)

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

    return np.where(largest_box >=1, True, False)


def cut_to_box(image, box_bounds, inverse=False, cropped_image=None, pad=0):

    box_boundary = box_bounds

    xlim_box = [box_boundary[0].start, box_boundary[0].stop]
    ylim_box = [box_boundary[1].start, box_boundary[1].stop]
    zlim_box = [box_boundary[2].start, box_boundary[2].stop]
    size = [xlim_box[1] - xlim_box[0], ylim_box[1] - ylim_box[0], zlim_box[1] - zlim_box[0]]
    size = [np.ceil(x).astype(int) for x in size]

    
    if inverse:

        assert np.product(image.shape) > np.product(cropped_image.shape)

        idx = np.zeros_like(image).astype(bool)
        returnimage = np.zeros_like(image).astype(float)

        idx[(xlim_box[0]-pad):(xlim_box[1]+pad), (ylim_box[0]-pad):(ylim_box[1]+pad), (zlim_box[0]-pad):(zlim_box[1]+pad)] = 1

        returnimage[idx] = cropped_image.flatten()

        return returnimage
    
    else:

        assert pad == 0
        assert cropped_image is None

        returnimage = np.zeros(tuple(size))

        returnimage = image[xlim_box[0]:xlim_box[1], ylim_box[0]:ylim_box[1], zlim_box[0]:zlim_box[1],]

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


def store_during_callback(current_iteration, hyperparameters, files, Jd, l2loss,
                            domainmesh, current_pde_solution, control):

    print_overloaded("Iter", format(current_iteration, ".0f"), 
                    "Jd =", format(Jd, ".4e"), 
                    "L2loss =", format(l2loss, ".4e")) 


    if MPI.rank(MPI.comm_world) == 0:
    
        with open(files["lossfile"], "a") as myfile:
            myfile.write(str(float(Jd))+ ", ")
        with open(files["l2lossfile"], "a") as myfile:
            myfile.write(str(float(l2loss))+ ", ")

    hyperparameters["Jd_current"] = float(Jd)
    hyperparameters["Jl2_current"] = float(l2loss)
    
    
    with XDMFFile(hyperparameters["outputfolder"] + "/State_checkpoint.xdmf") as xdmf:
        xdmf.write_checkpoint(current_pde_solution, "CurrentState", 0.)


    with XDMFFile(hyperparameters["outputfolder"] + "/Control_checkpoint.xdmf") as xdmf:
        xdmf.write_checkpoint(control, "CurrentV", 0.)


    file = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/CurrentControl.hdf", "w")
    file.write(control, "function")
    file.close()    

    # ROUTINE TO STORE TRANSFORMED BRAIN DURING OPTIMIZATION
    # COULD BE USEFUL FOR PLOTTING
    # DO NOT DELETE
    # if min(hyperparameters["input.shape"]) > 1 and len(hyperparameters["input.shape"]) == 3 and (current_iteration % 10 == 0):

    #     retimage = fem2mri(function=current_pde_solution, shape=hyperparameters["input.shape"])
        
    #     if MPI.rank(MPI.comm_world) == 0:
    #         nifti = nibabel.Nifti1Image(retimage, nibabel.load(hyperparameters["input"]).affine)

    #         nibabel.save(nifti, hyperparameters["outputfolder"] + '/mri/state_at_' + str(current_iteration) + '.mgz')

    #         print_overloaded("Stored mgz image of transformed image at iteration", current_iteration)
