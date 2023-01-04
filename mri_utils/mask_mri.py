import nibabel
import numpy as np
import os
# import matplotlib.pyplot as plt
import pathlib
import argparse
import json
from scipy import ndimage

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


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", type=str, nargs="+", required=True)
    parser.add_argument("--targetfolder", required=True, type=str)
    
    parser.add_argument("--crop", action="store_true", default=False)

    parser.add_argument("--coarsen", action="store_true", default=False)
    parser.add_argument("--npad", default=4, type=int)
    parser.add_argument("--zoom", default=0.5, type=float)


    parserargs = vars(parser.parse_args())

    assert parserargs["crop"] or parserargs["coarsen"]

    npad = parserargs["npad"]

    print(parserargs)

    targetfolder = pathlib.Path(parserargs["targetfolder"])

    os.makedirs(targetfolder, exist_ok=True)

    print("Will compute the largest bounding box for the following images:")

    for imagefile in parserargs["images"]:

        print("--", imagefile)

        if not os.path.isfile(imagefile):
            raise ValueError(imagefile + " does not exist")

    largest_box = get_largest_box(parserargs["images"])

    generic_affine = np.eye(4)

    np.save(targetfolder / "box.npy", largest_box)

    with open(targetfolder / "files.json", 'w') as outfile:
        json.dump(parserargs, outfile, sort_keys=True, indent=4)
        
    freeview_command = "freeview "

    for imgfile in parserargs["images"]:

        image = nibabel.load(imgfile).get_fdata()

        if parserargs["crop"]:
            image = cut_to_box(image, largest_box)
            outfile = str(targetfolder / ("cropped_" + pathlib.Path(imgfile).name))

        if parserargs["coarsen"]:

            padded_img = np.pad(image, npad, pad_with)

            image = ndimage.zoom(padded_img, parserargs["zoom"])

            print("Cropping images")

            outfile = str(targetfolder / ("coarsened" + pathlib.Path(imgfile).name))


        direc = pathlib.Path(imgfile).parent

        os.makedirs(direc, exist_ok=True)            
        
        print("saving to", outfile)

        freeview_command += outfile + " "

        

        nibabel.save(nibabel.Nifti1Image(image, affine=generic_affine), outfile)
    print("Final shape of images", image.shape)

    print()
    print(":"*100)
    print("MRI processing done, to view files in freeview, run")
    print()
    print(freeview_command)
    print()
    print(":"*100)