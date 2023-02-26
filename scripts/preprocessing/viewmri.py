import nibabel
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("-a", "--axis", type=int, default=1)
parser.add_argument("-i", "--idx", type=int, default=100)
parserarg = vars(parser.parse_args())

idx = parserarg["idx"]
axis = parserarg["axis"]

line = "freeview "

for img in parserarg["images"]:

    line += img + " "

    plt.figure()

    plt.title(img)

    print(img)
    img = nibabel.load(img)

    
    print(img.affine)
    print()

    if not isinstance(img, np.ndarray):
        img = img.get_fdata()

    if not (np.allclose(img.shape[axis], 256)):
        idx2 = int(img.shape[axis] * idx / 255)
    else:
        idx2 = idx

    plt.imshow(np.take(img, idx2, axis), cmap="Greys_r")# , vmax=100)
    # 
print(line)
plt.show()