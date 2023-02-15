import nibabel
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import argparse
import json


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", type=str, nargs="+", help="list of absolute file paths you want info about. Alternatively you can specify --imagedir")
    parser.add_argument("--imagedir", type=str, default=None, help="iterate over all files in this folder.  Alternatively you can specify --images")
    parser.add_argument("--nonzero", default=False, action="store_true", help="Print and plot info about all non-zero voxels")
    parserargs = vars(parser.parse_args())

    if parserargs["imagedir"] is not None:
        parserargs["images"] = []
        imdir = pathlib.Path(parserargs["imagedir"])
        for x in os.listdir(imdir):
            parserargs["images"].append(str(imdir / x))

    for imagefile in [x for x in parserargs["images"] if ".mgz" in x]:

        

        if not os.path.isfile(imagefile):
            raise ValueError(imagefile + " does not exist")


        image = nibabel.load(imagefile)

        image = image.get_fdata()

        if parserargs["nonzero"]:
            image = image[image > 0]


        imagename = pathlib.Path(imagefile).name
        print("--", imagefile)

        print("min=", np.min(image))
        print("max=", np.max(image))
        print("mean=", np.mean(image))
        print("median=", np.median(image))

        plt.figure()
        plt.hist(image.flatten())
        plt.yscale("log")
        plt.xlabel("Voxel Intensity")
        plt.ylabel("Count")
        plt.title(imagename)


    plt.show()
