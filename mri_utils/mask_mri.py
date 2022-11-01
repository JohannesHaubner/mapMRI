import nibabel
import numpy as np
import os
import matplotlib.pyplot as plt


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


def get_largest_box(pats):

    largest_box = np.zeros((256, 256, 256))


    for pat in pats:

        mfile = os.path.join(datapath, pat, "MASKS", "parenchyma.mgz")

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
    xlim = [box_boundary[0].start, box_boundary[0].stop]
    ylim = [box_boundary[1].start, box_boundary[1].stop]
    zlim = [box_boundary[2].start, box_boundary[2].stop]
    
    size = [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]]
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


datapath = "/home/basti/Dropbox (UiO)/Sleep/"

# filename = "mask_only"
filename = "masked"

pats = ["091", "205"]

ax = 2
slice_index = 120

slices = []



largest_box = get_largest_box(pats)

generic_affine = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ])



for pat in pats:

    cpath = os.path.join(datapath, pat, "CONFORM")
    cfile = os.path.join(cpath, sorted(os.listdir(cpath))[0])
    print(pat, cfile)
    mfile = os.path.join(datapath, pat, "MASKS", "parenchyma.mgz")
    aff = nibabel.load(mfile).affine
    mask = nibabel.load(mfile).get_fdata()# .astype(bool)

    if filename == "mask_only":
        img1 = mask

    elif filename == "masked":

        img1 = mask * nibabel.load(cfile).get_fdata()

        img1 = img1 / img1.max()


    img2 = cut_to_box(img1, largest_box)

    # breakpoint()

    nibabel.save(nibabel.Nifti1Image(img2, affine=generic_affine), pat + "_cropped.mgz")