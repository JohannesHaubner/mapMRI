import os
import nibabel
import numpy as np

datapath = "/home/basti/Dropbox (UiO)/Sleep/"
pats = ["091", "205"]

for pat in pats:


    mask = nibabel.load(os.path.join(datapath, pat, "mri"))