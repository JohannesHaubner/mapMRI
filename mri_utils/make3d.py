import numpy as np
import nibabel
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt


target_dir = "./"


N = 32

wx = 2 * 1
wy = 4 * 1
wz = 4 * 1

center = (N-1) / 2

input = np.zeros((N, N, N))

for x in range(N):

    if abs(x - center) > wx:
        continue

    for y in range(N):
        if abs(y - center) > wy:
            continue

        for z in range(N):
            if abs(z - center) > wz:
                continue

            input[x, y, z] = 1.

print(input.sum() / input.size)

input = scipy.ndimage.gaussian_filter(input, sigma=(1, 1, 1))

ds = 2
rotx = 1
dx, dy, dz = ds, ds, ds

aff = np.array([
    [  rotx,    0,    0,  dx],
    [   0,    1,    0,  dy],
    [   0,   0,    1,  dz],
    [   0,    0,    0,    1]
    ])

aff = np.linalg.inv(aff)


target = scipy.ndimage.affine_transform(input, aff)

print(input.sum(), target.sum())
idx, ax = center, 1

insl = np.take(input, idx, ax)
# insl = np.where(insl > 0, insl, np.nan)

outsl = np.take(target, idx, ax)
outsl = np.where(outsl > 1e-1, outsl, np.nan)

plt.imshow(insl, cmap="Greys")
plt.imshow(outsl, cmap="Reds")
plt.colorbar()
plt.show()

generic_affine = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ])

# input += 0.01
# target += 0.01


np.save(target_dir + "input.npy", input)
np.save(target_dir + "target.npy", target)

nibabel.save(nibabel.Nifti1Image(input, affine=generic_affine), target_dir + "input.mgz")
nibabel.save(nibabel.Nifti1Image(target, affine=generic_affine), target_dir + "target.mgz")

# aff205 = np.array([[  -1,    0,    0,  122],
#        [   0,    0,    1, -108],
#        [   0,   -1,    0,   57],
#        [   0,    0,    0,    1]])

# in_mri = nibabel.Nifti1Image(input, affine=aff205)

# 