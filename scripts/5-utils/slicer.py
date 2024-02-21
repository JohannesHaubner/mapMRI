import nibabel
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to .mgz file")
parser.add_argument("--axis", type=int, help="Axis along which to slice", choices=[0, 1, 2])
parser.add_argument("--pos", type=int, help="Index where to slice (typically between 50 and 200)",)
parser.add_argument("--output", type=str, help="Path to save the image to (optional).", default=None)
parser.add_argument("--bounds", type=int, nargs="+", help="Bounds to manually cut the image", default=None)

args = vars(parser.parse_args())

oimage = nibabel.load(args["image"]).get_fdata()

print("Shape of the original image: ", oimage.shape)

image_slice = np.take(oimage, axis=args["axis"], indices=args["pos"])

image_slice = np.where(image_slice < 0, 0, image_slice)
image_slice = np.where(np.isnan(image_slice), 0, image_slice)
image_slice = np.where(image_slice > 150, 150, image_slice)

if args["bounds"] is not None:
    image_slice = image_slice[args["bounds"][0]:args["bounds"][1], args["bounds"][2]:args["bounds"][3]]

print(args["image"])

plt.figure()
plt.imshow(image_slice, cmap="Greys_r")
# disable axis
plt.axis('off')
# remove white space
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()

if args["output"] is not None:
    plt.savefig(args["output"], dpi=300, bbox_inches="tight")
    plt.close()
else:
    plt.show()

