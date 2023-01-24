import scipy
import scipy.ndimage
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel
import itertools
from dgregister.helpers import crop_rectangular, get_bounding_box_limits
from tqdm import tqdm

if "bastian" in os.getcwd():
    p1 = "/home/bastian/D1/registration/"

else:
    p1 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation"


imagefiles = [
p1 + "/mri2fem-dataset/processed/affine_registered/abbytoernie_affine.mgz",
p1 + "/mri2fem-dataset/processed/affine_registered/ernie_brain.mgz",
]


imagefiles = [nibabel.load(x).get_fdata() for x in imagefiles]

imagefiles = [np.where(x>0, True, False) for x in imagefiles]

im1, im2 = imagefiles[0], imagefiles[1]
idx, ax = 100, 0
fig, (az1, az2, az3) = plt.subplots(1,3)
az1.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
idx, ax = 100, 1
az2.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
idx, ax = 100, 2
az3.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
plt.show()

boundary = get_bounding_box_limits(crop_rectangular(imagefiles))

xlim = [boundary[0].start, boundary[0].stop]
ylim = [boundary[1].start, boundary[1].stop]
zlim = [boundary[2].start, boundary[2].stop]

dx = xlim[1] - xlim[0]
dy = ylim[1] - ylim[0]
dz = zlim[1] - zlim[0]

dn = 2

n0 = (dx + 2 * dn) * (dy + 2 * dn) * (dz + 2 * dn)

print("Without rotation", n0, format(n0 / imagefiles[0].size, ".2f"), "of total voxels")

bestalpha, bestbeta, bestgamm = 0, 0, 0

r = R.from_euler('zyx', [bestalpha, bestbeta, bestgamm], degrees=True).as_matrix()
affine = np.zeros((4,4))
affine[:3, :3] = r
affine[-1,-1] = 1

im1 = scipy.ndimage.affine_transform(input=imagefiles[0], matrix=affine, order=3)
im2 = scipy.ndimage.affine_transform(input=imagefiles[1], matrix=affine, order=3)

idx, ax = 100, 0
fig, (az1, az2, az3) = plt.subplots(1,3)
az1.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
idx, ax = 100, 1
az2.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
idx, ax = 100, 2
az3.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
plt.show()

a = range(0, 10, 2)
b = range(0, 10, 2)
c = range(0, 10, 2)

minbox = 1e16
maxbox = 0
worst, best = None, None


progress = tqdm(total=len(a)*len(b)*len(c))

for (alpha, beta, gamma) in itertools.product(a,b,c):
    r = R.from_euler('zyx', [alpha, beta, gamma], degrees=True).as_matrix()
    affine = np.zeros((4,4))
    affine[:3, :3] = r

    affine[-1,-1] = 1

    im1 = scipy.ndimage.affine_transform(input=imagefiles[0], matrix=affine).astype(bool)
    im2 = scipy.ndimage.affine_transform(input=imagefiles[1], matrix=affine).astype(bool)


    im1 = np.round(scipy.ndimage.median_filter(input=im1, size=2), decimals=0)
    im2 = np.round(scipy.ndimage.median_filter(input=im2, size=2), decimals=0)


    try:
        boundary = get_bounding_box_limits(crop_rectangular([im1, im2]))
    except ValueError:
        print(alpha, beta, gamma, "yields value error")
        # idx, ax = 100, 0
        # fig, (az1, az2, az3) = plt.subplots(1,3)
        # az1.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
        # idx, ax = 100, 1
        # az2.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
        # idx, ax = 100, 2
        # az3.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
        # plt.show()

        continue

    xlim = [boundary[0].start, boundary[0].stop]
    ylim = [boundary[1].start, boundary[1].stop]
    zlim = [boundary[2].start, boundary[2].stop]

    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    dz = zlim[1] - zlim[0]

    dn = 2

    n = (dx + 2 * dn) * (dy + 2 * dn) * (dz + 2 * dn)

    # print(alpha, beta, gamma, n)

    if n > maxbox:
        maxbox = n
        worst = (alpha, beta, gamma)

    if n < minbox:
        minbox = n
        best = (alpha, beta, gamma)

    progress.update(1)

print("best: ", minbox, "voxels", format(n0 / minbox, ".2f"), "compared to no rotation")
print("worst:", maxbox, "voxels", format(n0 / maxbox, ".2f"), "compared to no rotation")

# Load cropped images
# loop over a itercools.product of angles
# find smallest bounding box
