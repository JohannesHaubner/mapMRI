from fenics import *
from fenics_adjoint import *
import os
import numpy
import numpy as np
import nibabel
import json
import pathlib
from parse import parse
import matplotlib.pyplot as plt

from nibabel.affines import apply_affine


class Data():

    def __init__(self, input, target) -> None:

        if (not "abby" in input) and (not "ernie" in target):  # "ventricle" in input or "hydrocephalus" in input:

            box = np.load("/home/bastian/D1/registration/hydrocephalus/freesurfer/021/testouts/box_all.npy")
            space = 2
            pad = 2

            aff3 = nibabel.load("/home/bastian/D1/registration/hydrocephalus/normalized/registered/021to068.mgz").affine
            # self.input_meshfile = "/home/bastian/D1/registration/hydrocephalus/meshes/ventricle_boundary.xml"
            # self.input_meshfile = "/home/bastian/D1/registration/hydrocephalus/mymeshes/021ventricles_boundary.xml"
            # self.input_meshfile = "/home/bastian/D1/registration/hydrocephalus/meshes/ventricles.xml"
            self.input_meshfile = "/home/bastian/D1/registration/hydrocephalus/meshes/ventricles_boundaryinvFalse.xml"
            
            self.original_target = "/home/bastian/D1/registration/hydrocephalus/" + "normalized/input/068/068_brain.mgz"
            self.original_input = "/home/bastian/D1/registration/hydrocephalus/" + "normalized/input/021/021_brain.mgz"
            self.registration_lta = "/home/bastian/D1/registration/hydrocephalus/" + "normalized/registered/021to068.lta" 


        else:
            assert "abby" in input
            assert "ernie" in target
            box = np.load("/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/box.npy")
            space = 0
            pad = 2

            aff3 = nibabel.load("/home/bastian/D1/registration/mri2fem-dataset/normalized/registered/abbytoernie.mgz").affine

            # self.registration_lta = "/home/bastian/D1/registration/mri2fem-dataset/" + "normalized/registered/abbytoernie.lta"

            self.input_meshfile = "/home/bastian/D1/registration/mri2fem-dataset/meshes/ventricles/abby/affreg-ventricle-boundarymesh.xml"
            self.original_input = "/home/bastian/D1/registration/mri2fem-dataset/normalized/registered/abbytoernie.mgz"
            ##  Alternative:
            ## Use the registration affine in meshtransport.
            ## TODO FIXME make sure the conversion from vox2vox is correct.
            ## (Be careful: freesurfer-RAS vs freesurfer-surface-RAS coordinates!!!)
            # self.input_meshfile = "/home/bastian/D1/registration/mri2fem-dataset/chp4/outs/abby/abby16.xml"
            # self.original_input = "/home/bastian/D1/registration/" + "mri2fem-dataset/normalized/input/abby/" + "abby_brain.mgz"
            ## this should then be needed / accessed:
            # self.target_meshfile = "/home/bastian/D1/registration/mri2fem-dataset/chp4/outs/ernie/ernie16.xml"

            self.original_target = "/home/bastian/D1/registration/" + "mri2fem-dataset/normalized/input/ernie/" + "ernie_brain.mgz"


            
        self.vox2ras_input = nibabel.load(self.original_input).header.get_vox2ras_tkr()
        self.vox2ras_target = nibabel.load(self.original_target).header.get_vox2ras_tkr()

        if hasattr(self, "registration_lta"):

            self.registration_affine = read_vox2vox_from_lta(self.registration_lta)

        self.inputmesh = Mesh(self.input_meshfile)

        self.box = box
        self.space = space
        self.pad = pad
        self.affine = aff3

        bounds = get_bounding_box_limits(self.box)
        self.dxyz = [bounds[x].start for x in range(3)]

    def meshcopy(self) -> Mesh:
        return Mesh(self.input_meshfile)


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass


def get_cras_from_nifti(image_nii):
    """Get the center of the RAS coordinate system.

    Based on: https://neurostars.org/t/freesurfer-cras-offset/5587/2
    Which links to: https://github.com/nipy/nibabel/blob/d1518aa71a8a80f5e7049a3509dfb49cf6b78005/nibabel/freesurfer/mghformat.py#L630-L643
    """
    shape = np.array(image_nii.shape)
    center = shape / 2
    center_homogeneous = np.hstack((center, [1]))
    transform = image_nii.affine
    return (transform @ center_homogeneous)[:3]


def get_surface_ras_to_image_coordinates_transform(image_nii, surface_metadata=None):
    """Convert from freesurfer surface coordinates to scanner coordinates.

    Freesurfer uses (at least) three different coordinate systems, the RAS system,
    the surface RAS system (which is a shifted version of the RAS system) and the
    image coordinates. This function creates a transformation matrix that transforms
    the surface RAS system into image coordinates. To accomplish this, it uses the
    cras (center of RAS) property from the surface metadata to translate the surface
    RAS into the correct RAS coordinates. Then it uses the inverse of the image
    coordinate to RAS transformation to transform the RAS coordinates into image
    coordinates.

    The surface metadata is obtained from using the ``nibabel.freesurfer.read_geometry``
    function like this:

        points, faces, metadata = nib.freesurfer.read_geometry(path, read_metadata=True)
    
    The image coordinate transform is the affine transform in the nifti file whose image
    coordinates we want to transform into.

    If the metadata is not supplied, then it is assumed to be created from an image with
    the same coordinate systems as the given nifti file.

    Example
    -------


    >>> import nibabel as nib
    ... points, faces, metadata = nib.freesurfer.read_geometry("lh.pial", read_metadata=True)
    ... img = nib.load("T2W.nii")
    ... translation_matrix = get_surface_ras_to_image_coordinates_transform(img, metadata)
    ... translation_matrix = get_surface_ras_to_image_coordinates_transform(img)
    """
    translation_matrix = np.eye(4)
    if surface_metadata is not None:
        translation_matrix[:3, -1] = surface_metadata['cras']
    else:
        translation_matrix[:3, -1] = get_cras_from_nifti(image_nii)

    return np.linalg.inv(image_nii.affine)@translation_matrix


def move_mesh(meshfile, ltafile, inverse, vox2ras, vox2ras2, allow_ras2ras):
    
    regaff = read_vox2vox_from_lta(ltafile, allow_ras2ras)
    bm = Mesh(meshfile)
    
    if inverse:
        regaff = numpy.linalg.inv(regaff)
    
    regaff = np.matmul((vox2ras2), np.matmul(regaff, np.linalg.inv(vox2ras)))

    xyz = np.copy(bm.coordinates()[:])
    # bm.coordinates()[:] = apply_affine(vox2ras, apply_affine(regaff, apply_affine(np.linalg.inv(vox2ras), xyz)))
    bm.coordinates()[:] = apply_affine(regaff, xyz)

    return bm



def crop_to_original(orig_image: numpy.ndarray, cropped_image: numpy.ndarray, box: numpy.ndarray, space: int, pad: int):

    box_bounds = get_bounding_box_limits(box)

    # if space == 2:
    #     assert "abby" not in original_image_path
    #     assert "ernie" not in original_image_path

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



def view(images, axis, idx, colorbar=False):

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


def cut_to_box(image, box_bounds, inverse=False, cropped_image=None, pad=0):

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

        assert np.product(image.shape) > np.product(cropped_image.shape)

        # raise NotImplementedError

        idx = np.zeros_like(image).astype(bool)
        returnimage = np.zeros_like(image).astype(float)

        # idx[image_center[0] - int(size[0] / 2):image_center[0] + int(size[0] / 2),
        #             image_center[1] - int(size[1] / 2):image_center[1] + int(size[1] / 2),
        #             image_center[2] - int(size[2] / 2):image_center[2] + int(size[2] / 2),
        # ] = 1

        # breakpoint()
        idx[(xlim_box[0]-pad):(xlim_box[1]+pad), (ylim_box[0]-pad):(ylim_box[1]+pad), (zlim_box[0]-pad):(zlim_box[1]+pad)] = 1

        returnimage[idx] = cropped_image.flatten()

        return returnimage
    
    else:

        assert pad == 0
        assert cropped_image is None

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


def store_during_callback(current_iteration, hyperparameters, files, Jd, l2loss,
                            domainmesh, current_pde_solution, control):

    print_overloaded("Iter", format(current_iteration, ".0f"), 
                    "Jd =", format(Jd, ".4e"), 
                    "L2loss =", format(l2loss, ".4e")) # , "Reg =", format(Jreg, ".4e"))



    if MPI.rank(MPI.comm_world) == 0:
    
        with open(files["lossfile"], "a") as myfile:
            myfile.write(str(float(Jd))+ ", ")
        # with open(files["regularizationfile"], "a") as myfile:
        #     myfile.write(str(float(Jreg))+ ", ")
        with open(files["l2lossfile"], "a") as myfile:
            myfile.write(str(float(l2loss))+ ", ")
        # with open(files["totallossfile"], "a") as myfile:
        #     myfile.write(str(float(Jd + Jreg))+ ", ")

    hyperparameters["Jd_current"] = float(Jd)
    hyperparameters["Jl2_current"] = float(l2loss)
    
    
    with XDMFFile(hyperparameters["outputfolder"] + "/State_checkpoint.xdmf") as xdmf:
        xdmf.write_checkpoint(current_pde_solution, "CurrentState", 0.)


    with XDMFFile(hyperparameters["outputfolder"] + "/Control_checkpoint.xdmf") as xdmf:
        xdmf.write_checkpoint(control, "CurrentV", 0.)

    # file = HDF5File(domainmesh.mpi_comm(), hyperparameters["outputfolder"] + "/CurrentV.hdf", "w")
    # file.write(velocityField, "function")
    # file.close()        


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
