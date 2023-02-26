import nibabel
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from dgregister.helpers import view

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("-a", "--axis", type=int, default=1)
parser.add_argument("-i", "--idx", type=int, default=100)
parserarg = vars(parser.parse_args())

idx = parserarg["idx"]
axis = parserarg["axis"]



view(parserarg["images"])
