import scipy
import scipy.ndimage
import scipy.optimize
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel
import itertools
from dgregister.helpers import crop_rectangular, get_bounding_box_limits, cut_to_box, pad_with
from tqdm import tqdm
import json
if "bastian" in os.getcwd():
    p1 = "/home/bastian/D1/registration/"

else:
    p1 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation"


imagenames = [
p1 + "/mri2fem-dataset/processed/affine_registered/abbytoernie_affine.mgz",
p1 + "/mri2fem-dataset/processed/affine_registered/ernie_brain.mgz",
]

table = np.genfromtxt("rotation.txt", delimiter=",")

bestidx = np.argmin(table[:,-1])

best = table[bestidx, :]

print("best", best)

bestalpha, bestbeta, bestgamm = best[0], best[1], best[2]



imagefiles = [nibabel.load(x).get_fdata() for x in imagenames]

# imagefiles = [np.where(x>0, True, False) for x in imagefiles]

# im1, im2 = imagefiles[0], imagefiles[1]
# idx, ax = 100, 0
# fig, (az1, az2, az3) = plt.subplots(1,3)
# az1.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
# idx, ax = 100, 1
# az2.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
# idx, ax = 100, 2
# az3.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
# plt.show()

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



r = R.from_euler('zyx', [bestalpha, bestbeta, bestgamm], degrees=True).as_matrix()
affine = np.zeros((4,4))
affine[:3, :3] = r
affine[-1,-1] = 1



# mri = nibabel.Nifti1Image(im1.astype(float), np.eye(4))
# nibabel.save(mri, "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/affine-rotated/testabby.mgz")

# bi1 = np.round(scipy.ndimage.median_filter(input=im1, size=2), decimals=0)
# mri = nibabel.Nifti1Image(bi1.astype(float), np.eye(4))
# nibabel.save(mri, "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/affine-rotated/testabby2.mgz")

# exit()

im1 = scipy.ndimage.affine_transform(input=imagefiles[0], matrix=affine, order=3)
im2 = scipy.ndimage.affine_transform(input=imagefiles[1], matrix=affine, order=3)

bi1 = scipy.ndimage.affine_transform(input=imagefiles[0].astype(bool), matrix=affine, order=3).astype(bool)
bi2 = scipy.ndimage.affine_transform(input=imagefiles[1].astype(bool), matrix=affine, order=3).astype(bool)
bi1 = np.round(scipy.ndimage.median_filter(input=bi1, size=2), decimals=0)
bi2 = np.round(scipy.ndimage.median_filter(input=bi2, size=2), decimals=0)

boundary = get_bounding_box_limits(crop_rectangular([bi1, bi2]))

xlim = [boundary[0].start, boundary[0].stop]
ylim = [boundary[1].start, boundary[1].stop]
zlim = [boundary[2].start, boundary[2].stop]



dx = xlim[1] - xlim[0]
dy = ylim[1] - ylim[0]
dz = zlim[1] - zlim[0]

dn = 2

n = (dx + 2 * dn) * (dy + 2 * dn) * (dz + 2 * dn)
print("With best ", n, format(n / n0, ".2f"), "compared to before")




im1 = np.where(im1 < 0, 0, im1)
im1 = np.round(im1, decimals=0)

im2 = np.where(im2 < 0, 0, im2)
im2 = np.round(im2, decimals=0)


cutted_1 = cut_to_box(im1, boundary, inverse=False, cropped_image=None)
cutted_2 = cut_to_box(im2, boundary, inverse=False, cropped_image=None)

npad = 2

padded_img1 = np.pad(cutted_1, npad, pad_with)
padded_img2 = np.pad(cutted_2, npad, pad_with)

mri1 = nibabel.Nifti1Image(padded_img1, np.eye(4))
mri2 = nibabel.Nifti1Image(padded_img2, np.eye(4))

nibabel.save(mri1, "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/affine-rotated/abby.mgz")
nibabel.save(mri2, "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/affine-rotated/ernie.mgz")

np.savetxt("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/affine-rotated/best_affine.txt", affine, delimiter=",")
np.savetxt("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/affine-rotated/best_alpha_beta_gamma.txt", np.array([bestalpha, bestbeta, bestgamm]), delimiter=",")

data = {
    "imagefiles":imagenames,
    "npad": npad,
    "bestalpha, bestbeta, bestgamma": [bestalpha, bestbeta, bestgamm],
    "boundary": str(boundary), 
    }

with open("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/affine-rotated/parameters.json", 'w') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=4)





breakpoint()
exit()


# idx, ax = 100, 0
# fig, (az1, az2, az3) = plt.subplots(1,3)
# az1.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
# idx, ax = 100, 1
# az2.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
# idx, ax = 100, 2
# az3.imshow(np.take(im1, idx, ax)+np.take(im2, idx, ax))
# plt.show()

a = range(0, 20, 2)
b = range(0, 20, 2)
c = range(0, 10, 2)

minbox = 1e16
maxbox = 0
worst, best = None, None


progress = tqdm(total=len(a)*len(b)*len(c))

for (alpha, beta, gamma) in itertools.product(a,b,c):
    break
# def rot(x):

#     alpha, beta, gamma = x[0], x[1], x[2]

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
        # return 1e16

    xlim = [boundary[0].start, boundary[0].stop]
    ylim = [boundary[1].start, boundary[1].stop]
    zlim = [boundary[2].start, boundary[2].stop]

    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    dz = zlim[1] - zlim[0]

    dn = 2

    n = (dx + 2 * dn) * (dy + 2 * dn) * (dz + 2 * dn)

    # print(alpha, beta, gamma, n)

    # if n > maxbox:
    #     maxbox = n
    #     worst = (alpha, beta, gamma)

    # if n < minbox:
    #     minbox = n
    #     best = (alpha, beta, gamma)

    with open("rotation.txt", "a") as myfile:
        myfile.write(str(alpha) + "," + str(beta) + "," + str(gamma) + "," + str(n))
        myfile.write("\n")

    # return n 
    progress.update(1)




# res = scipy.optimize.minimize(fun=rot, x0=[0,0,0], args=(), method="L-BFGS-B", options={"iprint":101})

# print(res)

print("best: ", minbox, "voxels", format(n0 / minbox, ".2f"), "compared to no rotation")
print(best)
print("worst:", maxbox, "voxels", format(n0 / maxbox, ".2f"), "compared to no rotation")
print(worst)
# Load cropped images
# loop over a itercools.product of angles
# find smallest bounding box
