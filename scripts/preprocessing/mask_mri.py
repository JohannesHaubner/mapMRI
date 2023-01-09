import nibabel
import numpy as np
import os
# import matplotlib.pyplot as plt
import pathlib
import argparse
import json
from scipy import ndimage
from dgregister.helpers import get_largest_box, pad_with, cut_to_box, get_bounding_box


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