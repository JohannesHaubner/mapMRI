import nibabel
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to .mgz file")
parser.add_argument("--axis", type=int, help="Axis along which to slice", choices=[0, 1, 2])
parser.add_argument("--pos", type=int, help="Index where to slice (typically between 50 and 200)",)
parser.add_argument("--output", type=str, help="Path to save the image to (optional).", default=None)

args = vars(parser.parse_args())

oimage = nibabel.load(args["image"]).get_fdata()

image_slice = np.take(oimage, axis=args["axis"], indices=args["pos"])

plt.figure()
plt.imshow(image_slice, cmap="Greys_r")
# disable axis
plt.axis('off')
# remove white space
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()

if args["output"]:
    plt.savefig(args["output"], dpi=300, bbox_inches="tight")
# plt.show()

plt.close()